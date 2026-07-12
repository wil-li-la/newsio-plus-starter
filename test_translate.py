#!/usr/bin/env python3
"""Newsio+ 翻譯測試腳本 — 載入基底模型 + 你訓練好的 LoRA adapter，翻一則新聞。

用法：
  python test_translate.py                      # 用內建範例新聞
  python test_translate.py --title "..." --bullets "..." "..."
  python test_translate.py --no-adapter         # 對照組：看「沒微調」的原始模型表現
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 必須和 train_lora.py 用同一句 system prompt，模型才會進入「翻譯模式」。
SYSTEM_PROMPT = (
    "你是 Newsio 的新聞翻譯編輯。"
    "將英文新聞標題與重點翻譯成台灣慣用的繁體中文，保持 Newsio 簡潔風格。"
)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR = os.path.join("out", "adapter")

# 內建範例（沒給 --title 時使用）
SAMPLE_TITLE = "OpenAI releases GPT-5.2 with improved multilingual reasoning"
SAMPLE_BULLETS = [
    "The new model scores 30% higher on non-English benchmarks.",
    "API pricing stays unchanged from GPT-5.1.",
    "Enterprise customers get early access starting next week.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Newsio+ 翻譯測試腳本")
    parser.add_argument("--title", type=str, default=SAMPLE_TITLE, help="英文新聞標題")
    parser.add_argument("--bullets", type=str, nargs="+", default=SAMPLE_BULLETS,
                        help="英文重點，可以給多個（用空白分隔的多個字串）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"基底模型，預設 {DEFAULT_MODEL}（要和訓練時相同）")
    parser.add_argument("--adapter", type=str, default=ADAPTER_DIR,
                        help=f"adapter 路徑，預設 {ADAPTER_DIR}")
    parser.add_argument("--no-adapter", action="store_true",
                        help="不載入 adapter，看原始模型的表現（對照組）")
    return parser.parse_args()


def load_model(args):
    """載入基底模型（有 GPU 就用 4-bit 量化，省 VRAM），再視情況疊上 adapter。"""
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto"
        )
    else:
        # 沒有 GPU：用 CPU 慢慢跑也行（3B 模型推論還算可行，只是要等一下）。
        print("（沒偵測到 CUDA GPU，改用 CPU 推論 — 會比較慢）")
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    if not args.no_adapter:
        if not os.path.isdir(args.adapter):
            raise SystemExit(
                f"找不到 adapter：{args.adapter}\n"
                "請先跑 python train_lora.py 訓練，或加 --no-adapter 看原始模型表現。"
            )
        # 把 LoRA adapter 疊到基底模型上 — 這就是你訓練出來的那 0.29%。
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    return model


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args)

    user_text = args.title.strip() + "\n" + "\n".join(f"- {b}" for b in args.bullets)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    # add_generation_prompt=True → 在結尾加上 assistant 起始標記，請模型接著寫翻譯。
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    print("\n=== 英文原文 ===")
    print(user_text)
    print("\n=== 繁體中文翻譯 ===")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            do_sample=False,          # 翻譯任務用 greedy decoding，結果穩定可重現
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只解碼「新生成」的部分（去掉 prompt）。
    generated = output_ids[0][input_ids.shape[-1]:]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())
    print()


if __name__ == "__main__":
    main()
