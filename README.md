# Newsio+ Starter — 用 LoRA 微調 Llama 3.2 3B 做新聞翻譯

這是 [Newsio+](https://plus.newsio.io) 課程的起手式（starter）程式庫。

## 這是什麼

Newsio 每天用 AI 把英文科技新聞整理成標題與重點摘要。這個 repo 教你：

> **用 LoRA 微調 Llama 3.2 3B，讓它學會把 Newsio 的英文新聞標題與重點，翻譯成台灣慣用的繁體中文。**

你不需要重新訓練整個 3B 參數的模型。LoRA 只在 attention 的
q_proj / k_proj / v_proj / o_proj 上掛一組小小的 adapter（rank r = 16 時，
每個 transformer block 只多 r × 20,480 = 327,680 個參數，28 個 block 加起來
9,175,040 個 —— 大約只佔整個模型的 0.29%），一張 8GB 的 GPU 或免費的
Colab T4 就能訓練。

課程網站（原理講解、互動教材）：**https://plus.newsio.io**

## 需求

- Python 3.10 以上
- NVIDIA GPU（VRAM 8GB 以上，訓練用 4-bit QLoRA）
  - 沒有 GPU？直接用免費的 [Google Colab（T4）](#colab)
- 磁碟空間約 10GB（模型權重 + 資料）

## 快速開始

```bash
# 1. 下載這個 repo
git clone https://github.com/wil-li-la/newsio-plus-starter.git
cd newsio-plus-starter

# 2. 建立虛擬環境並安裝套件
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. 把 email 收到的 train.jsonl 放進 data/
#    （怎麼拿到資料？見下一節）
mv ~/Downloads/train.jsonl data/

# 4. 開始訓練（預設 r=16、1 個 epoch）
python train_lora.py
```

想調整超參數：

```bash
python train_lora.py --rank 8 --epochs 2 --batch 1 --lr 1e-4 --max-samples 2000
```

## 資料哪裡來

到 **https://plus.newsio.io/data.html** 用 Google 帳號登入，按「寄送資料」，
`train.jsonl` 的下載連結會寄到你的信箱（連結 7 天內有效）。
下載後放到這個 repo 的 `data/` 資料夾。

## 資料格式

`train.jsonl` 共 **14,746 筆**，每行一筆 JSON，欄位如下：

| 欄位 | 說明 |
| --- | --- |
| `news_id` | Newsio 文章 id |
| `category` | 分類（genai、semiconductor、robotics…） |
| `en_title` | 英文標題（模型的輸入） |
| `en_bullets` | 英文重點列表（模型的輸入） |
| `zh_title` | 繁體中文標題（模型要學的輸出） |
| `zh_bullets` | 繁體中文重點列表（模型要學的輸出） |
| `source_name` | 原始新聞來源名稱 |
| `source_url` | 原始新聞連結 |

訓練時，`train_lora.py` 會把每筆資料組成 Llama 3.2 的 chat 格式：
system 是固定的翻譯編輯指令，user 是英文標題 + 重點，assistant 是繁中標題 + 重點。
損失（loss）只算在 assistant 的部分 —— 模型只需要學「怎麼翻」，不用學「怎麼出題」。

## 訓練參數說明

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--rank` | 16 | LoRA 的 rank（r）。r 越大 adapter 越大、學得越多也越容易 overfit。可選 1–64。alpha 固定是 2×r。 |
| `--epochs` | 1 | 整份資料看幾遍。14,746 筆看 1 遍通常就夠了。 |
| `--batch` | 2 | 每張 GPU 一次吃幾筆。VRAM 不夠就調成 1（腳本會用 gradient accumulation 補回有效 batch size）。 |
| `--lr` | 2e-4 | 學習率。LoRA 常用 1e-4 ~ 3e-4。 |
| `--max-samples` | 全部 | 只用前 N 筆訓練，想快速實驗時很好用。 |
| `--model` | unsloth/Llama-3.2-3B-Instruct | 基底模型。也可以改成 meta-llama/Llama-3.2-3B-Instruct（需要先在 Hugging Face 申請存取權）。 |

訓練完成後，adapter 會存在 `out/adapter/`（只有幾十 MB —— 這就是 LoRA 的好處）。

## 訓練完之後

用 `test_translate.py` 試翻一則新聞：

```bash
# 用內建的範例新聞
python test_translate.py

# 或自己給標題與重點
python test_translate.py \
  --title "Nvidia unveils next-gen Rubin GPU platform" \
  --bullets "The chip doubles inference throughput over Blackwell." \
            "Mass production is slated for late 2026."
```

腳本會載入基底模型 + 你剛訓練好的 `out/adapter/`，印出繁體中文翻譯。
想對比「微調前 vs 微調後」，加上 `--no-adapter` 就能看原始模型的表現。

## Colab

沒有 GPU 的話，用免費的 Colab T4 就能跑完整流程：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wil-li-la/newsio-plus-starter/blob/main/colab_train.ipynb)

notebook：[`colab_train.ipynb`](colab_train.ipynb) —— 安裝套件、上傳 train.jsonl、
用 r=16 訓練、試翻一則新聞、打包下載 adapter，一格一格照著跑即可。

## 授權

程式碼以 [MIT License](LICENSE) 釋出 © 2026 Newsio。
訓練資料（train.jsonl）僅供教學用途，請勿再散布。
