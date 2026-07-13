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
  - **比較新的顯示卡（RTX 40／50 系列）**：要裝夠新、CUDA build 相符的 PyTorch
    （例如 `cu12x`／`cu13`）。裝到太舊的 torch，訓練時會出現
    `CUDA error: no kernel image is available for execution on the device`。
- 磁碟空間**約 15GB**：torch + CUDA 套件約 6GB、基底模型約 6.5GB，另外 pip
  下載時預設還會佔用約 5GB 的快取。空間很緊的話，安裝時加 `--no-cache-dir`
  （見下方），避免快取把硬碟塞爆、導致模型只下載一半而壞掉。

## 快速開始

```bash
# 1. 下載這個 repo
git clone https://github.com/wil-li-la/newsio-plus-starter.git
cd newsio-plus-starter

# 2. 建立虛擬環境並安裝套件
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --no-cache-dir -r requirements.txt   # --no-cache-dir 省下約 5GB 快取

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

## 先跑跑看：內建範例資料

還沒拿到 email 的完整資料，也可以先用 repo 裡附的 **`data/sample.jsonl`**
（10 筆、涵蓋 genai／semiconductor／fintech）跑一遍，確認環境沒問題：

```bash
python train_lora.py --data data/sample.jsonl --max-samples 10 --val-frac 0
```

這一步能在等 email 之前就先驗證：PyTorch + CUDA 在你的機器上跑得起來（尤其
RTX 40／50 系列）、基底模型（約 6.5GB）下載沒壞掉、tokenizer 的 chat template
在你的 transformers 版本正常、整個訓練迴圈至少能前進一步。10 筆資料當然訓練
不出好模型 —— 它只是用來 smoke-test。

## 資料哪裡來（完整 14,746 筆）

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
| `--val-frac` | 0.05 | 切多少比例當**驗證集**：每個 epoch 會印一次 `eval_loss`，用來判斷有沒有 overfit。設 `0` 可關掉。 |
| `--keep-empty-bullets` | 關 | 預設會**丟掉 `zh_bullets` 為空的資料**（原始資料約 12% 是空的，留著會教模型「有時候不用輸出重點」）。加這個旗標可保留全部資料。 |
| `--model` | unsloth/Llama-3.2-3B-Instruct | 基底模型。也可以改成 meta-llama/Llama-3.2-3B-Instruct（需要先在 Hugging Face 申請存取權）。 |

> **資料清理與驗證集**：預設 `train_lora.py` 會先濾掉沒有繁中重點的資料，再切
> 5% 出來當驗證集。訓練時每個 epoch 會看到 `train` 和 `eval` 兩個 loss —
> 如果 `train_loss` 一直降但 `eval_loss` 開始回升，就是 overfit 了。
> 想在**沒看過的資料**上算翻譯品質（chrF、繁簡、重點數…），用 `eval_translate.py`。

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

### 用數字驗證：`eval_translate.py`

只看一則翻譯不夠客觀。`eval_translate.py` 會對一批新聞分別用「基底模型」和
「基底模型 + adapter」翻譯，再和資料裡真正的繁中翻譯比對，算出幾個指標：

```bash
python eval_translate.py --n 200            # 從 data/train.jsonl 抽 200 筆比較
```

| 指標 | 意義 | 方向 |
| --- | --- | --- |
| `chrF` | 和參考翻譯的字元 n-gram 相似度（中文比 BLEU 合適） | 越高越好 |
| 重點數一致 | 產生的重點數量和參考是否相同 | 越高越好 |
| 簡體字比例 | 不小心吐簡體字的比例（我們要 zh-TW 繁體） | 越低越好 |
| 英文殘留字數 | 沒翻到的英文（GPT／API 這種專有名詞保留是正常的，只當參考） | 越低越好 |

想要**乾淨的類推分數**（模型沒看過的資料），先切一份 hold-out 出來，訓練時避開它，
再用 `--data <holdout.jsonl>` 評估。

## Colab

沒有 GPU 的話，用免費的 Colab T4 就能跑完整流程：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wil-li-la/newsio-plus-starter/blob/main/colab_train.ipynb)

notebook：[`colab_train.ipynb`](colab_train.ipynb) —— 安裝套件、上傳 train.jsonl、
用 r=16 訓練、試翻一則新聞、打包下載 adapter，一格一格照著跑即可。

## 授權

- **程式碼**：以 [MIT License](LICENSE) 釋出 © 2026 Newsio。
- **訓練資料**（`train.jsonl`）：僅供教學用途，請勿再散布。
- **基底模型 Llama 3.2**：受 [Meta Llama 3.2 社群授權](https://www.llama.com/llama3_2/license/)
  規範（**不是** MIT／Apache 這種完全開放的授權）。不管你用哪個 HF 鏡像，使用權重就等於同意它。重點：
  - 研究與商業用途都免費；只有**月活躍用戶超過 7 億**的公司才需要另外向 Meta 申請授權。
  - 「不可在歐盟使用」的限制**只適用於多模態（vision）版本**；本課程用的 **1B／3B 純文字模型在歐盟可以正常使用**。
  - 再散布模型時要標註「Built with Llama」並附上授權檔。

### 需要下載模型的存取權嗎？

- **預設的 `unsloth/Llama-3.2-3B-Instruct`（本課程用的）**：**不用**。這個鏡像沒有 gated，
  不需要 Hugging Face 帳號、token 或申請核准，`train_lora.py` 直接就能下載。
- **官方的 `meta-llama/Llama-3.2-3B-Instruct`（用 `--model` 才會用到）**：**要**。官方 repo 是
  gated 的，步驟如下：
  1. 到 [模型頁面](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) 登入 Hugging Face、
     填表接受 Meta 的授權，等核准（Llama 3.2 通常很快）。
  2. 到 [Settings → Access Tokens](https://huggingface.co/settings/tokens) 產生一組 token。
  3. 本機登入：`huggingface-cli login`（或設環境變數 `export HF_TOKEN=hf_xxx`）。
  4. 再跑 `python train_lora.py --model meta-llama/Llama-3.2-3B-Instruct`。
