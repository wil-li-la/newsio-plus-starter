#!/usr/bin/env python3
"""Newsio+ 驗證腳本 — 用數字比較「微調前 vs 微調後」的翻譯品質。

對一批新聞，分別用「基底模型」和「基底模型 + LoRA adapter」翻譯，然後和資料裡
真正的繁中參考翻譯比對，算出幾個指標：

  chrF         — 字元層級的 n-gram F-score（0~100，越高越接近參考翻譯；中文比 BLEU 合適）
  bullet_match — 產生的重點數量是否和參考一致（例如都是 5 點）
  simp_rate    — 輸出裡出現「簡體字」的比例（越低越好，我們要的是繁體 zh-TW）
  ascii_words  — 輸出裡長度≥4 的英文字串平均數量（像 "早期-access" 這種沒翻到的殘留；
                 GPT/OpenAI/API 這類專有名詞保留是正常的，所以這個只當參考）

⚠️ 注意：train_lora.py 是「全部 14,746 筆」都拿去訓練的，沒有預留驗證集。
   所以這裡抽樣到的資料「模型訓練時看過」，chrF 偏高是正常的 —
   它比較能說明「有沒有學到 Newsio 的風格對應」，而不是純粹的「對新資料的類推能力」。
   想要乾淨的類推分數，要用 --holdout：只評估「訓練時被跳過的尾端資料」
   （搭配 train_lora.py --max-samples <N> 一起用）。
"""

import argparse
import collections
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "你是 Newsio 的新聞翻譯編輯。"
    "將英文新聞標題與重點翻譯成台灣慣用的繁體中文，保持 Newsio 簡潔風格。"
)
DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_DIR = os.path.join("out", "adapter")
DATA_PATH = os.path.join("data", "train.jsonl")

# 常見「只在簡體用」的字（用來偵測模型不小心吐簡體）。不是完整表，但足以當訊號。
SIMPLIFIED_ONLY = set("开发释产运动电报华应众亿万与后台价类过还这来对应门问间时长东车马鸟龙"
                      "语说读书爱国关战权归汇团队实现历标签网络优质")


def as_lines(value):
    if isinstance(value, str):
        value = [value]
    return "\n".join(f"- {x}".strip() for x in value if str(x).strip())


def ngram_counts(s, n):
    s = s.replace(" ", "").replace("\n", "")
    return collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp, ref, maxn=6, beta=2.0):
    precisions, recalls = [], []
    for n in range(1, maxn + 1):
        h, r = ngram_counts(hyp, n), ngram_counts(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / max(1, sum(h.values())))
        recalls.append(overlap / max(1, sum(r.values())))
    if not precisions:
        return 0.0
    P = sum(precisions) / len(precisions)
    R = sum(recalls) / len(recalls)
    if P + R == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * P * R / (b2 * P + R)


def bullet_count(text):
    return sum(1 for ln in text.splitlines() if ln.strip().startswith(("-", "•", "‧")))


def simp_char_rate(text):
    han = [c for c in text if "一" <= c <= "鿿"]
    if not han:
        return 0.0
    return sum(1 for c in han if c in SIMPLIFIED_ONLY) / len(han)


def ascii_word_count(text):
    return len(re.findall(r"[A-Za-z]{4,}", text))


def load_records(path, n, holdout):
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    # 只評估「有繁中重點」的資料，比較公平（空 bullets 的無法比對重點）。
    recs = [r for r in recs if (r.get("zh_bullets") or []) and (r.get("en_bullets") or [])]
    if holdout:
        return recs[-n:]                 # 尾端 N 筆（搭配 --max-samples 就是沒訓練到的）
    # 固定間隔抽樣（可重現，不用 random）
    step = max(1, len(recs) // n)
    return recs[::step][:n]


def build_prompt(tok, rec):
    user = rec["en_title"].strip() + "\n" + as_lines(rec["en_bullets"])
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt")
    return enc


def generate(model, tok, enc):
    enc = enc.to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"],
                             attention_mask=enc.get("attention_mask"),
                             max_new_tokens=512, do_sample=False,
                             repetition_penalty=1.05,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][enc["input_ids"].shape[-1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def evaluate(model, tok, records, tag):
    rows = []
    for i, rec in enumerate(records):
        ref = rec["zh_title"].strip() + "\n" + as_lines(rec["zh_bullets"])
        hyp = generate(model, tok, build_prompt(tok, rec))
        rows.append(dict(
            chrf=chrf(hyp, ref),
            bmatch=1.0 if bullet_count(hyp) == bullet_count(ref) else 0.0,
            simp=simp_char_rate(hyp),
            ascii=ascii_word_count(hyp),
        ))
        print(f"  [{tag}] {i+1}/{len(records)} chrF={rows[-1]['chrf']:.1f}", flush=True)
    n = len(rows)
    agg = lambda k: sum(r[k] for r in rows) / n
    return dict(n=n, chrF=agg("chrf"), bullet_match=agg("bmatch"),
                simp_rate=agg("simp"), ascii_words=agg("ascii"))


def load_base(model_name):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=(torch.bfloat16
                                 if torch.cuda.is_bf16_supported() else torch.float16))
    return AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                                device_map="auto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=ADAPTER_DIR)
    ap.add_argument("--n", type=int, default=40, help="評估幾筆新聞")
    ap.add_argument("--data", default=DATA_PATH, help=f"評估資料，預設 {DATA_PATH}")
    ap.add_argument("--holdout", action="store_true",
                    help="只評估尾端 N 筆（搭配 train --max-samples 才是真正沒看過的資料）")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    records = load_records(args.data, args.n, args.holdout)
    print(f"評估 {len(records)} 筆（holdout={args.holdout}）\n")

    print("=== 微調前（基底模型）===")
    base = load_base(args.model)
    base.eval()
    before = evaluate(base, tok, records, "before")
    del base
    torch.cuda.empty_cache()

    print("\n=== 微調後（+ LoRA adapter）===")
    m = load_base(args.model)
    m = PeftModel.from_pretrained(m, args.adapter)
    m.eval()
    after = evaluate(m, tok, records, "after")

    print("\n" + "=" * 56)
    print(f"{'指標':<16}{'微調前':>12}{'微調後':>12}{'變化':>12}")
    print("-" * 56)
    def row(label, k, pct=False, up_good=True):
        b, a = before[k], after[k]
        d = a - b
        arrow = ("✅" if (d > 0) == up_good else "⚠️ ") if abs(d) > 1e-6 else "  "
        fmt = (lambda x: f"{x*100:.1f}%") if pct else (lambda x: f"{x:.2f}")
        print(f"{label:<16}{fmt(b):>12}{fmt(a):>12}{fmt(d):>11} {arrow}")
    row("chrF (↑好)", "chrF")
    row("重點數一致 (↑好)", "bullet_match", pct=True)
    row("簡體字比例 (↓好)", "simp_rate", pct=True, up_good=False)
    row("英文殘留字數", "ascii_words", up_good=False)
    print("=" * 56)


if __name__ == "__main__":
    main()
