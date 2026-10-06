# Nova LLM — 从零训练的 8M 参数迷你 MoE 对话模型

纯 PyTorch 从零实现的微型大语言模型（MiniMoE 架构）：字符/词混合 tokenizer → 预训练 → LoRA SFT → CPU 本地推理，全程不依赖任何预训练权重。仓库内附带训练好的权重，克隆后即可直接对话。

## 模型架构（约 8M 参数）

| 组件 | 配置 |
|---|---|
| 层数 / 隐藏维度 | 4 层 Transformer decoder，dim=256 |
| 注意力 | 8 头 GQA（4 个 KV 头），RMSNorm 前置 |
| FFN | SwiGLU，中间维度 512 |
| MoE | 4 专家 top-2 路由 |
| 上下文长度 | 192 token |
| 微调 | LoRA（r=16, alpha=32，注入 qkvo + MoE 投影） |
| Tokenizer | 自研字符 + 高频词混合词级 tokenizer，含对话/工具调用特殊 token |

## 快速开始

```bash
pip install -r requirements.txt
python chat.py
```

- 强制 CPU 推理，无需 GPU。
- 对话命令：`quit` 退出，`/reset` 清空历史。
- 支持多轮对话、基本算术（`what is 2 plus 3` 直接规则解析，模型也可发出 `<tool>` 计算器工具调用）、基础语法问答与脏话拒答。

## 仓库结构

```
├── config.py               # 8M MoE + LoRA 全局配置
├── model.py                # MiniMoE 模型定义（含 apply_lora）
├── vocab.py                # 字符/词混合 tokenizer
├── chat.py                 # CPU 本地推理（多轮对话 + 工具调用）
├── build_sft_english.py    # 程序化生成英文 SFT 数据集
├── validate_sft.py         # 用项目 tokenizer 校验 SFT 数据格式与长度
├── sft_english_custom.jsonl# 生成的 SFT 数据（多轮对话 / tool call / 语法教学）
├── checkpoints/
│   ├── pretrain.pt         # 预训练权重
│   ├── lora.pt             # LoRA SFT 权重
│   └── tokenizer.json      # tokenizer 词表
└── minimoe/                # 原始版本：113M 参数中文版，含完整训练流水线
```

`minimoe/` 是本项目的原始版本（dim=512、6 层、8 专家、全量 SFT 的中文变体），保留了完整的训练代码，根目录的 8M LoRA 英文版由它演化而来。

## 训练（使用 minimoe/ 流水线）

```bash
cd minimoe
bash run_all.sh   # 安装依赖 → build_data.py 生成数据 → train_pretrain.py → train_sft.py
```

或分步执行：`python build_data.py`（程序化生成预训练与 SFT 语料）→ `python train_pretrain.py` → `python train_sft.py` → `python chat.py`。

## 重新生成 SFT 数据（根目录英文版）

```bash
python build_sft_english.py   # 输出 sft_english_custom.jsonl
python validate_sft.py        # 校验长度 ≤ MAX_SEQ_LEN(192) 与格式
```

## 已知限制

- 8M 参数的模型能力非常有限，回复偏模板化，仅用于学习 MoE / GQA / LoRA / tokenizer / CPU 推理的完整流程。
- 算术主要靠规则与工具调用兜底，模型本身的计算不可靠。
- 历史窗口仅保留最近 2 轮对话。
