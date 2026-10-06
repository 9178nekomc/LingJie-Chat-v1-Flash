# validate_sft.py — 用项目实际 tokenizer 逻辑验证 SFT 数据长度与格式
import json, sys, importlib.util

DATA = r"D:\Users\Coffee\Desktop\Nova LLM\sft_english_custom.jsonl"
VOCAB = r"D:\Users\Coffee\Desktop\Nova LLM\vocab.py"
MAX = 192

# 注入最小 config 桩，加载项目 vocab.py
import types
cfg = types.ModuleType("config")
setattr(cfg, "TOKENIZER_PATH", "")
for k, v in {"PAD_TOKEN": "<pad>", "BOS_TOKEN": "<s>", "EOS_TOKEN": "</s>", "UNK_TOKEN": "<unk>",
             "IM_START": "<|im_start|>", "IM_END": "<|im_end|>", "USER_TOKEN": "<|user|>",
             "ASSISTANT_TOKEN": "<|assistant|>", "TOOL_START": "<tool>", "TOOL_END": "</tool>",
             "ARGS_START": "<args>", "ARGS_END": "</args>"}.items():
    setattr(cfg, k, v)
sys.modules["config"] = cfg
spec = importlib.util.spec_from_file_location("vocab_mod", VOCAB)
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)

over = 0
bad_role = 0
empty = 0
tool_samples = []
lengths = []
with open(DATA, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        obj = json.loads(line)
        convs = obj["conversations"]
        # 格式检查
        roles = [m["role"] for m in convs]
        if not all(r in ("user", "assistant") for r in roles):
            bad_role += 1
        if any(not m["content"].strip() for m in convs):
            empty += 1
        # 长度检查（模拟 SFTDataset._encode 的 token 数）
        n = 2  # <s> + </s>
        for m in convs:
            n += 2 + len(vm.split_text(m["content"])) + 1  # im_start+role, content, im_end
        lengths.append(n)
        if n > MAX:
            over += 1
        if any("<tool>" in m["content"] for m in convs) and len(tool_samples) < 3:
            tool_samples.append((n, convs))

total = idx + 1
print(f"total conversations: {total}")
print(f"bad role: {bad_role} | empty content: {empty} | over MAX_SEQ_LEN(192): {over}")
import statistics
print(f"len: min={min(lengths)} max={max(lengths)} avg={statistics.mean(lengths):.1f}")

print("\n=== tool call samples ===")
for n, c in tool_samples:
    print(f"--- len={n} ---")
    for m in c:
        print(f"  [{m['role']}] {m['content'][:130]}")

print("\n=== random samples ===")
import random
random.seed(1)
with open(DATA, encoding="utf-8") as f:
    lines = f.readlines()
for i in random.sample(range(total), 6):
    convs = json.loads(lines[i])["conversations"]
    print(f"--- idx={i} ---")
    for m in convs:
        print(f"  [{m['role']}] {m['content'][:120]}")
