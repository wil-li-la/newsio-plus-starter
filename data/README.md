# data/ — 訓練資料放這裡

把你 email 收到的 `train.jsonl` 放進這個資料夾：

```
data/train.jsonl
```

## 怎麼取得資料

1. 到 **https://plus.newsio.io/data.html**
2. 用 Google 帳號登入，按「寄送資料」
3. `train.jsonl`（14,746 筆 EN→zh-TW 訓練配對，約 23MB）的下載連結會寄到你的信箱
4. 連結 7 天內有效 — 下載後放進這個資料夾即可

> 注意：`.gitignore` 已設定忽略 `data/*.jsonl`，訓練資料不會（也不應該）被 commit 進 git。
> 資料僅供教學用途，請勿再散布。
