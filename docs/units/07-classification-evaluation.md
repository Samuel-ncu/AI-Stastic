# 分類模型評估

[在 Colab 執行](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/07_classification_evaluation.ipynb) · [GitHub Notebook](../../notebooks/07_classification_evaluation.ipynb)

## 學習目標與先備知識

先讀第二課的條件機率。學完能自行列出混淆矩陣、手算指標，並依漏判及誤報成本選擇要關注的結果。

## 1. 先定義正類，再計算

**來源摘要 [S6](https://scikit-learn.org/stable/modules/model_evaluation.html)：**Precision 與 Recall 分別以「預測正類」和「實際正類」作為分母；F1 以調和平均彙整兩者。多類別的平均方式會改變整體指標的意義。

正類不代表「好的事」。在本教材自編檢測案例中，1=缺陷，0=良品。先寫清楚這件事，才能知道什麼是漏判。

| 實際＼預測 | 預測良品 0 | 預測缺陷 1 |
|---|---|---|
| 實際良品 0 | TN：正確放行 | FP：誤報、可能多做檢查 |
| 實際缺陷 1 | FN：漏判、可能流出 | TP：正確攔下 |

本課矩陣採列為真實、欄為預測。閱讀不同網站圖表時先看軸標籤，不能只靠位置記憶 TP。

## 2. 用同一張表逐步手算

本教材自編 1,000 件產品，990 良品、10 缺陷。候選模型 TP=8、FN=2、FP=20、TN=970。

$$Accuracy=\frac{TP+TN}{N}=\frac{978}{1000}=97.8\%$$

$$Precision=\frac{TP}{TP+FP}=\frac8{28}\approx28.57\%$$

「系統叫你檢查的 28 件，有多少真的缺陷？」這是 Precision。

$$Recall=\frac{TP}{TP+FN}=\frac8{10}=80\%$$

「真正 10 件缺陷，有多少被抓到？」這是 Recall。

$$F1=\frac{2TP}{2TP+FP+FN}=\frac{16}{38}\approx42.11\%$$

F1 不是 Precision 與 Recall 的普通算術平均，也不直接包含 TN。

## 3. 為什麼 99% 可能比較沒用？

永遠猜良品的基準能得到 990/1000=99% Accuracy，但 Recall=0。它漏掉全部缺陷。本例候選模型 Accuracy 較低，卻抓到 8 件缺陷。

若假設每個 FN 成本 100、FP 成本 1，基準成本為 10×100=1,000；候選成本為 2×100+20=220。這些成本是教學假設，真實專案需向利害關係人確認。

## 4. 閾值如何連到決策？

模型輸出分數後，要指定何時判為正類。降低閾值通常會標出更多正類，同時也可能增加誤報。應在驗證資料比較，而不是看測試資料挑出最好看的閾值。

不同任務也不一定以 F1 為最佳目標。例如若漏判有硬性上限，可先要求 Recall 達標，再比較誤報數或成本。

## 5. 延伸到 DLI 多類別手語

逐類查看誰容易和誰混淆。Macro-F1 對每個類別的 F1 等權平均；Weighted-F1 依各類實際樣本數加權，所以多數類仍可能影響較大。不要只報一個平均而省略類別表現。

如果模型完全不預測正類，Precision 的原分母為 0。程式設 zero_division=0 是回報規則，不表示數學上原本就能做 0/0。

## 自我檢核

1. TP=6、FP=4、FN=2 時，Precision 與 Recall 各是多少？
2. 多數類占 99%，永遠猜多數類是否證明模型學會有用規律？
3. F1 高是否保證誤判總成本最低？

### 參考解答

1. Precision=6/10=60%；Recall=6/8=75%。
2. 沒有；應與其他基準及每類結果比較。
3. 不保證。F1 沒有直接表達所有成本，也不包含 TN。

## 實作與延伸作業

請開啟上方 Notebook，逐格執行程式，對照手算與輸出。完成 Notebook 末尾的修改練習，繳交結果圖表、解讀、至少一項限制及 AI 使用驗證紀錄。

## 出處與閱讀建議

- **S6**：[scikit-learn，Metrics and scoring（Log loss、Precision、Recall）](https://scikit-learn.org/stable/modules/model_evaluation.html)。

本課以所列來源的概念作繁體中文摘要，並加入自行編寫的情境、手算、程式及練習；不是原網站的逐段翻譯。標為「來源摘要」者為改寫，標為「自編」者為本教材教學設計；短引文另以引號及「中譯」標明。來源連結可供對照原文，查閱日期：2026-11-17。
