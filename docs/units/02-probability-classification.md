# 機率與分類

[在 Colab 執行](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/02_probability_classification.ipynb) · [GitHub Notebook](../../notebooks/02_probability_classification.ipynb)

[背景導讀：從統計到 AI 的模型設計](../statistics-to-ai.md)

## 學習目標與先備知識

先複習比例與百分比。學完能區分「有缺陷時驗出陽性」與「驗出陽性時有缺陷」，並理解分類分數為何需要驗證。

## 1. 從人數開始理解機率

**來源摘要 [O4](https://openstax.org/books/introductory-statistics-2e/pages/3-1-terminology)：**事件是可能結果的集合。機率位於 0 到 1；以等可能的有限結果計算時，可用符合事件的結果數除以全部結果數。

**本教材自編例子：**盒內有 80 個良品、20 個缺陷品，等機率抽一個。令 D 表示「抽到缺陷品」，則 P(D)=20/100=0.2。0.2 是 20%，不是每連抽五個就保證有一個缺陷品。

| 寫法 | 意思 |
|---|---|
| $P(D)$ | 缺陷事件的機率 |
| $\neg D$ | 非缺陷 |
| $D\cap+$ | 有缺陷，而且被判陽性 |
| $P(D\mid+)$ | 已知被判陽性後，有缺陷的機率 |

## 2. 條件機率：分母到底換成哪一群？

**來源摘要 [O5](https://openstax.org/books/introductory-statistics-2e/pages/3-3-two-basic-rules-of-probability)：**$P(A\mid B)=P(A\cap B)/P(B)$，前提是 $P(B)>0$。條件機率把觀察範圍限制在 B 發生的情況。

看清楚直線右邊的條件：P(+|D) 的分母只看缺陷品；P(D|+) 的分母只看陽性判定。兩者問不同問題，不能交換。

## 3. 用 10,000 件產品推導貝氏計算

以下為本教材自編、機制已知的假想檢測，不是真實醫療或產品數據。假設缺陷率 1%，對缺陷品的檢出率 90%，對良品的偽陽性率 5%。

| 真實狀態 | 判陽性 | 判陰性 | 合計 |
|---|---:|---:|---:|
| 缺陷 | 90 | 10 | 100 |
| 良品 | 495 | 9,405 | 9,900 |
| 合計 | 585 | 9,415 | 10,000 |

先不背公式，只看表：陽性有 585 件，其中 90 件真的缺陷。因此 P(D|+)=90/585≈15.38%。90% 的檢出率沒有變，但良品太多，仍產生不少偽陽性。

將同一計算用機率表示，就是：

$$P(D\mid+)=\frac{P(+\mid D)P(D)}
{P(+\mid D)P(D)+P(+\mid\neg D)P(\neg D)}$$

分子是「真缺陷且判陽性」；分母是「所有判陽性的可能途徑」。此式由條件機率與把陽性事件分解為缺陷／非缺陷兩部分得到。

## 4. 從模型分數到機率

神經網路可先輸出未正規化分數 $z_k$，再經 softmax：

$$p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}$$

$k$ 是類別，分母加總所有類別，因此輸出加總為 1。下方程式減去最大的 z 是數值穩定技巧，不改變數學上的比例。

**來源摘要 [S7](https://scikit-learn.org/stable/modules/calibration.html)：**校準關心預測機率與實際發生比例是否一致。例如正類機率約為 0.8 的一群樣本，若模型校準良好，其中約 80% 應為正類。

這是「一群樣本」的可靠性，不是替單張影像提供保證。DLI 手語模型即使最高分 0.99，也可能遇到不在訓練分布的手勢。Accuracy 高與機率校準良好，也不是同一件事。

### 補充｜機率如何連到訓練損失？

用「是否超時」為例，模型先輸出一個 logit，Sigmoid 再把它轉成超時機率 $p$。未超時的機率是 $1-p$。**二元交叉熵（BCE）取實際答案對應機率的負自然對數。**

| 實際答案 | 若模型預測超時機率為 0.8，真實答案分到多少？ | 單筆損失 |
|---|---:|---:|
| 超時 | $p=0.8$ | $-\ln(0.8)\approx0.223$ |
| 未超時 | $1-p=0.2$ | $-\ln(0.2)\approx1.609$ |

如果任務改成在「正常、超時、失敗」中單選，Softmax 把各類分數轉成總和為 1 的機率，再用多類別交叉熵。如果一筆請求可能同時超時與辨識錯誤，則針對每個標籤各用一個 Sigmoid 和一項 BCE。兩種交叉熵都使用對數。把同一批樣本的逐筆損失相加，是從概似推導訓練目標的做法。

若實際超時，兩個模型分別給 0.8 和 0.6，也都可能在閾值 0.5 下答對，但交叉熵分別約為 0.223 和 0.511。至於分數有沒有校準，還需看一批獨立資料中「預測約 0.8」的請求實際有多少比例超時。

[第 05 課：交叉熵通式、BCE 推導與取對數的手算](05-statistical-models-loss.md) · [來源：PyTorch 交叉熵](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) · [scikit-learn 機率校準](https://scikit-learn.org/stable/modules/calibration.html)

## 5. 程式前先做預測

把缺陷率由 1% 改成 10%，保持其他檢測機制不變。真陽性與偽陽性各會怎樣變？先畫表，再執行下方程式。

## 自我檢核

1. P(+|D)=0.9 能否直接寫成 P(D|+)=0.9？
2. 缺陷率 10%、檢出率 90%、偽陽性率 5% 時，陽性後缺陷機率是多少？
3. softmax 加總為 1，是否已證明分數校準？

### 參考解答

1. 不能，條件不同。
2. $0.09/(0.09+0.045)=2/3$，約 66.67%。
3. 沒有；仍需獨立資料檢查機率與觀察比例。

## 實作與延伸作業

請開啟上方 Notebook，逐格執行程式，對照手算與輸出。完成 Notebook 末尾的修改練習，繳交結果圖表、解讀、至少一項限制及 AI 使用驗證紀錄。

## 出處與閱讀建議

- **O4**：[OpenStax，§3.1 Terminology](https://openstax.org/books/introductory-statistics-2e/pages/3-1-terminology)。
- **O5**：[OpenStax，§3.3 Two Basic Rules of Probability](https://openstax.org/books/introductory-statistics-2e/pages/3-3-two-basic-rules-of-probability)。
- **S7**：[scikit-learn，Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)。

本課以所列來源的概念作繁體中文摘要，並加入自行編寫的情境、手算、程式及練習；不是原網站的逐段翻譯。標為「來源摘要」者為改寫，標為「自編」者為本教材教學設計；短引文另以引號及「中譯」標明。來源連結可供對照原文，查閱日期：2026-11-17。
