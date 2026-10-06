# local_chat.py — 本地 CPU 推理
import os
import re
import torch

# 强制用 CPU
device = torch.device("cpu")
print(f"Device: {device}")

from config import (PRETRAIN_CKPT, LORA_CKPT, TOKENIZER_PATH,
                    BOS_TOKEN, EOS_TOKEN, IM_START, IM_END,
                    USER_TOKEN, ASSISTANT_TOKEN,
                    LORA_R, LORA_ALPHA, LORA_TARGETS)
from vocab import Tokenizer
from model import MiniMoE, apply_lora


KEEP_HISTORY = True
MAX_HISTORY_TURNS = 2
BAD_WORDS = ["fuck", "shit", "bitch", "damn", "bastard", "asshole", "crap"]


def try_direct_math(text):
    """直接从输入提取数学，绕过模型"""
    t = text.lower().strip()
    m = re.match(r"what is ([\d.]+)\s*plus\s*([\d.]+)", t)
    if m:
        return f"{m.group(1)} plus {m.group(2)} equals {float(m.group(1)) + float(m.group(2))}."
    m = re.match(r"what is ([\d.]+)\s*minus\s*([\d.]+)", t)
    if m:
        return f"{m.group(1)} minus {m.group(2)} equals {float(m.group(1)) - float(m.group(2))}."
    m = re.match(r"what is ([\d.]+)\s*times\s*([\d.]+)", t)
    if m:
        return f"{m.group(1)} times {m.group(2)} equals {float(m.group(1)) * float(m.group(2))}."
    m = re.match(r"what is ([\d.]+)\s*divided by\s*([\d.]+)", t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if b == 0:
            return "cannot divide by zero."
        return f"{m.group(1)} divided by {m.group(2)} equals {a / b}."
    return None


def calc_tool(args):
    try:
        args = args.replace(" ", "").strip()
        parts = args.split("|")
        if len(parts) != 3:
            return None
        a_s, op, b_s = parts
        if not a_s or not b_s:
            return None
        a = float(a_s) if "." in a_s else int(a_s)
        b = float(b_s) if "." in b_s else int(b_s)
        if op == "+": result, word = a + b, "plus"
        elif op == "-": result, word = a - b, "minus"
        elif op == "*": result, word = a * b, "times"
        elif op == "/":
            if b == 0: return "cannot divide by zero."
            result, word = a / b, "divided by"
        else: return None
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{a_s} {word} {b_s} equals {result}."
    except Exception:
        return None


def parse_tool_call(reply):
    compact = reply.replace(" ", "")
    m = re.search(r"<tool>calc</tool><args>(.*?)</args>", compact)
    return m.group(1) if m else None


def main():
    print("Loading tokenizer...")
    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    print(f"Vocab: {len(tokenizer.vocab)}")

    print("Loading model...")
    model = MiniMoE(vocab_size=len(tokenizer.vocab))
    ckpt = torch.load(PRETRAIN_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded pretrain")

    model = apply_lora(model, r=LORA_R, alpha=LORA_ALPHA,
                       dropout=0.0, targets=LORA_TARGETS)
    if os.path.exists(LORA_CKPT):
        lora_ckpt = torch.load(LORA_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(lora_ckpt["model"], strict=False)
        print(f"Loaded LoRA")

    model = model.to(device).eval()
    print("Ready. Commands: quit / reset")
    print("-" * 60)

    history = []
    eos_id = tokenizer.stoi[EOS_TOKEN]
    im_end_id = tokenizer.stoi[IM_END]

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() == "quit":
            break
        if user == "/reset":
            history = []
            print("[history cleared]")
            continue

        user_is_bad = any(w in user.lower() for w in BAD_WORDS)

        math_ans = try_direct_math(user)
        if math_ans:
            print(f"AI : {math_ans}")
            if not user_is_bad:
                history.append({"role": "user", "content": user})
                history.append({"role": "assistant", "content": math_ans})
                if len(history) > MAX_HISTORY_TURNS * 2:
                    history = history[-MAX_HISTORY_TURNS * 2:]
            continue

        ids = [tokenizer.stoi[BOS_TOKEN]]
        if KEEP_HISTORY and history:
            for msg in history[-MAX_HISTORY_TURNS * 2:]:
                ids.append(tokenizer.stoi[IM_START])
                ids.append(tokenizer.stoi[USER_TOKEN if msg["role"] == "user" else ASSISTANT_TOKEN])
                ids.extend(tokenizer.encode(msg["content"]))
                ids.append(tokenizer.stoi[IM_END])
        ids.append(tokenizer.stoi[IM_START])
        ids.append(tokenizer.stoi[USER_TOKEN])
        ids.extend(tokenizer.encode(user))
        ids.append(tokenizer.stoi[IM_END])
        ids.append(tokenizer.stoi[IM_START])
        ids.append(tokenizer.stoi[ASSISTANT_TOKEN])

        idx = torch.tensor([ids], dtype=torch.long, device=device)

        import time
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.3, top_k=20)
        dt = time.time() - t0

        gen_ids = out[0, len(ids):].tolist()
        cut = len(gen_ids)
        for i, tid in enumerate(gen_ids):
            if tid == im_end_id or tid == eos_id:
                cut = i
                break
        gen_ids = gen_ids[:cut]

        raw_reply = tokenizer.decode(gen_ids)
        print(f"[DEBUG] {raw_reply!r}  ({dt:.1f}s)")

        args = parse_tool_call(raw_reply)
        if args:
            reply = calc_tool(args) or raw_reply
        else:
            reply = raw_reply

        print(f"AI : {reply}")

        if user_is_bad:
            history = []
        else:
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": reply})
            if len(history) > MAX_HISTORY_TURNS * 2:
                history = history[-MAX_HISTORY_TURNS * 2:]


if __name__ == "__main__":
    main()