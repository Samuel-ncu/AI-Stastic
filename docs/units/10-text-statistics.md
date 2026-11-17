# 文字資料的統計

[在 Colab 執行](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/10_text_statistics.ipynb) · [GitHub Notebook](../../notebooks/10_text_statistics.ipynb)

## 學習目標與先備知識

先讀機率分布與分類。本課從數詞開始理解文字如何變成向量，再介紹熵及相似度；不要求先會語言模型。

## 1. 文字怎麼變成模型輸入？

**來源摘要 [S8](https://scikit-learn.org/stable/modules/feature_extraction.html)：**詞袋表示把文字切成 token、計數並形成數值特徵。單詞詞袋不保留詞序，因此有明顯限制。

本教材自編兩句「ai helps students」與「students learn ai」。建立詞彙表 [ai,helps,learn,students] 後：

| 句子 | ai | helps | learn | students |
|---|---:|---:|---:|---:|
| ai helps students | 1 | 1 | 0 | 1 |
| students learn ai | 1 | 0 | 1 | 1 |

每列是一個文件向量。若對所有文字先建立詞彙表再切分，測試資料資訊可能進入前處理；正式分類要在訓練文字 fit，再對其他集合 transform。本課的小表只演示表示法，沒有做測試效能評估。

中文不一定有空白分詞。切詞、字元 n-gram 或模型 tokenizer 是不同選擇，會影響後續頻率及向量，不能直接假設英文示範可原封不動套用。

## 2. 詞頻如何變成機率分布？

本教材自編文件「ai ai data model」，總詞數 4。p(ai)=2/4，p(data)=p(model)=1/4。三個比例加總為 1。

這個分布是「在此文件中隨機抽一個詞位置」的經驗詞頻分布，不是某個語言模型對下一個詞的預測分布。先說清楚隨機的是什麼，熵才有意義。

## 3. 熵：平均有多少不確定性？

**來源摘要 [SC2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)：**Shannon 熵以 $-\sum p_i\log p_i$ 定義；對數底數決定單位。SciPy 可依指定底數計算，也可計算兩個分布的相對熵。

使用底數 2 時：

$$H=-\sum_i p_i\log_2 p_i$$

本教材手算 [0.5,0.25,0.25]：

- 第一項：−0.5×(−1)=0.5。
- 另外兩項各為 −0.25×(−2)=0.5。
- 總和 H=1.5 bits。

若只有一種詞，p=1，H=0。若四種詞等比例，H=2 bits。這些值描述指定分布的不確定性，**不等於文章品質、知識價值或作者創意**。對 p=0 的項，按極限慣例記為 0，不直接計算 0×log(0)。

## 4. 餘弦相似度是怎麼算的？

對非零向量 u、v：

$$\cos(u,v)=\frac{u\cdot v}{\|u\|\|v\|}$$

點積是對應元素相乘後加總，長度是平方和開根號。本教材自編 u=[1,1,0]、v=[1,0,1]：點積為 1，兩者長度都是 $\sqrt2$，所以相似度為 0.5。

它比較方向。將同一計數向量全部乘 2，相似度仍為 1，不代表文件內容完全相同。零向量的數學公式分母為 0，實作需要明確處理。

## 5. 如何連到 DLI NLP？

先把「token、詞彙表、序列、向量」分清楚，再看 embedding 與序列模型。詞袋可當輕量基準，但「狗追貓」與「貓追狗」若只看相同詞的次數，可能得到相同表示；語意角色已被省略。

本課 entropy 是詞頻熵。第五課分類交叉熵是用真實標籤評估預測機率；語言模型下一詞交叉熵又有序列條件。公式有關，但資料、機率與評估問題不同。

## 自我檢核

1. 分布 [0.5,0.5] 的熵是多少 bits？
2. u=[1,0] 與 v=[2,0] 的餘弦相似度是多少？
3. 兩句話詞袋一樣，就能斷定語意一樣嗎？

### 參考解答

1. 1 bit。
2. 1；兩者方向相同。
3. 不能。詞序、語法及語境可能不同。

## 實作與延伸作業

請開啟上方 Notebook，逐格執行程式，對照手算與輸出。完成 Notebook 末尾的修改練習，繳交結果圖表、解讀、至少一項限制及 AI 使用驗證紀錄。

## 出處與閱讀建議

- **S8**：[scikit-learn，Feature extraction（Bag of Words）](https://scikit-learn.org/stable/modules/feature_extraction.html)。
- **SC2**：[SciPy，scipy.stats.entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)。

本課以所列來源的概念作繁體中文摘要，並加入自行編寫的情境、手算、程式及練習；不是原網站的逐段翻譯。標為「來源摘要」者為改寫，標為「自編」者為本教材教學設計；短引文另以引號及「中譯」標明。來源連結可供對照原文，查閱日期：2026-11-17。
