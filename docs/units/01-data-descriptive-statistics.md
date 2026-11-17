# 資料與描述統計

[在 Colab 執行](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/01_descriptive_statistics.ipynb) · [GitHub Notebook](../../notebooks/01_descriptive_statistics.ipynb)

## 學習目標與先備知識

只需要會加減乘除、平方與開根號。完成後應能回答：一筆資料代表誰？資料集中在哪裡？差異有多大？為什麼 AI 訓練前要先看資料？

## 1. 先知道自己量的是什麼

**來源摘要 [P1](https://online.stat.psu.edu/stat200/Lesson01)、[O1](https://openstax.org/books/introductory-statistics-2e/pages/1-1-definitions-of-statistics-probability-and-key-terms)：**統計先界定觀察對象、變數與欲推論的範圍，再蒐集及分析資料。樣本的數值摘要稱為統計量，母體相對應的數值稱為參數。

**本教材自編情境：**你想知道某款 AI 鍵盤的辨識延遲。把「指定韌體、指定測試條件下，未來所有按鍵操作」當作研究母體，今天量到的五次操作則是樣本。若今天只測一台裝置，不能直接代表所有使用者與所有裝置。

| 術語 | 在本例的意思 |
|---|---|
| 觀測值 observation | 一次操作量到的延遲 |
| 變數 variable | 延遲，單位 ms |
| 樣本數 n | 實際量了幾次 |
| 母體 population | 想回答問題的全部操作範圍 |
| 樣本 sample | 實際蒐集到的部分操作 |
| 特徵 feature | 模型輸入，例如像素、按壓速度 |
| 標籤 label | 想辨識的答案，例如按鍵種類 |

數字形式不代表一定是數量。「貓=0、狗=1」只是類別編碼；算出平均 0.6 不能解讀成一種介於貓與狗之間的動物。若明確定義 1=狗，這個平均才可另外解讀為狗的比例。

## 2. 中心：平均數與中位數各回答什麼？

**來源摘要 [O2](https://openstax.org/books/introductory-statistics-2e/pages/2-5-measures-of-the-center-of-the-data)、[N1](https://www.itl.nist.gov/div898/handbook/eda/section3/eda351.htm)：**平均數使用所有觀測值；中位數使用排序後的中間位置，因此對極端數值的敏感程度不同。

平均數可想成把總量平均分攤：

$$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i$$

| 符號 | 讀法與用途 |
|---|---|
| $x_i$ | 第 i 筆數值 |
| $n$ | 數值總筆數 |
| $\sum$ | 把指定範圍的數值全部相加 |
| $\bar{x}$ | x bar，樣本平均數 |

**自編手算：**五次延遲是 8、9、10、11、12 ms。總和 50，除以 5，平均數是 10 ms；排序後第三個值是 10，中位數也是 10 ms。

若最後一筆改成 62，平均數變成 $(8+9+10+11+62)/5=20$ ms，中位數仍是 10 ms。這不表示要刪掉 62；它可能是真實的偶發卡頓。若產品在意最壞情況，除了中心還要看尾端延遲。

## 3. 分散：為什麼還需要變異數與標準差？

**來源摘要 [O3](https://openstax.org/books/introductory-statistics-2e/pages/2-7-measures-of-the-spread-of-the-data)、[N2](https://itl.nist.gov/div898/handbook/eda/section3/eda356.htm)：**分散程度描述數值離中心有多遠。變異數用平方偏差彙整差異；標準差取平方根，回到原來的量測單位。

平均都是 10，不代表體驗相同。A 裝置每次都是 10 ms；B 裝置有時 6 ms、有時 14 ms。A 比較穩定。這是本教材用來連結產品判讀的例子。

$$s^2=\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1},\qquad s=\sqrt{s^2}$$

以 8、9、10、11、12 為例：

| 延遲 $x_i$ | 偏差 $x_i-10$ | 偏差平方 |
|---:|---:|---:|
| 8 | −2 | 4 |
| 9 | −1 | 1 |
| 10 | 0 | 0 |
| 11 | 1 | 1 |
| 12 | 2 | 4 |
| 合計 | 0 | 10 |

1. 若直接把偏差相加，正負會抵消，所以改看平方。
2. 樣本變異數 $s^2=10/(5-1)=2.5$，單位為 ms²。
3. 樣本標準差 $s=\sqrt{2.5}\approx1.581$ ms。

$n-1$ 與什麼有關？估計平均數後，五個偏差加總必須為 0；知道前四個，第五個就被決定了。在獨立同分布且變異數有限的條件下，這個分母使樣本變異數成為母體變異數的不偏估計。這個敘述針對變異數，**不表示開根號後的標準差也完全不偏**。

若只想描述眼前這五筆完整資料，也可以除以 $n$，得到描述性變異數 2、標準差約 1.414。Python 中 np.std(x) 預設除以 n；np.std(x, ddof=1) 除以 n−1。作業要交代使用哪個定義。

## 4. 分布與圖表：不要只報一個平均

**來源摘要 [N3](https://www.itl.nist.gov/div898/handbook/eda/section3/histogra.htm)：**直方圖把數值範圍分箱，呈現每個區間的筆數，可觀察中心、分散、偏斜及多峰。

以下像素實驗中，A=[100,100,100,100,100]，B=[0,50,100,150,200]，平均都為 100。A 的樣本標準差是 0；B 的偏差平方和是 25,000，標準差為 $\sqrt{25000/4}\approx79.057$。執行程式前，先預測哪張直方圖較寬。

分位數是排序後的位置摘要。中位數是第 50 百分位，Q1、Q3 分別對應第 25、75 百分位；IQR=Q3−Q1。小樣本分位數有不同插值慣例，軟體與手算不一定完全一致，要說明方法。

## 5. 怎麼連到 DLI？

在 MNIST 或 ASL 中，先數每類張數，再檢查像素分布。例如訓練影像普遍明亮、部署影像普遍昏暗，是值得調查的落差，不能單憑平均亮度就宣稱模型一定失敗。影像直方圖也不保留像素位置；把像素打亂，分布可完全相同，內容卻改變。

## 自我檢核

1. 資料 2、4、6 的平均、中位數與樣本標準差是多少？
2. 為什麼不能把三種手語的編碼平均後當作「平均手語」？
3. 兩個資料集標準差相同，是否表示它們的分布相同？

### 參考解答

1. 平均與中位數均為 4；偏差平方和為 8，除以 2 得變異數 4，樣本標準差為 2。
2. 編碼代表類別身份，沒有可平均的數量尺度。
3. 否。中心、尾端、峰數及其他結構仍可能不同。

## 實作與延伸作業

請開啟上方 Notebook，逐格執行程式，對照手算與輸出。完成 Notebook 末尾的修改練習，繳交結果圖表、解讀、至少一項限制及 AI 使用驗證紀錄。

## 出處與閱讀建議

- **P1**：[Penn State，STAT 200 Lesson 1: Collecting Data](https://online.stat.psu.edu/stat200/Lesson01)。
- **O1**：[OpenStax，Introductory Statistics 2e §1.1](https://openstax.org/books/introductory-statistics-2e/pages/1-1-definitions-of-statistics-probability-and-key-terms)。
- **O2**：[OpenStax，§2.5 Measures of the Center of the Data](https://openstax.org/books/introductory-statistics-2e/pages/2-5-measures-of-the-center-of-the-data)。
- **O3**：[OpenStax，§2.7 Measures of the Spread of the Data](https://openstax.org/books/introductory-statistics-2e/pages/2-7-measures-of-the-spread-of-the-data)。
- **N1**：[NIST/SEMATECH，§1.3.5.1 Measures of Location](https://www.itl.nist.gov/div898/handbook/eda/section3/eda351.htm)。
- **N2**：[NIST/SEMATECH，§1.3.5.6 Measures of Scale](https://itl.nist.gov/div898/handbook/eda/section3/eda356.htm)。
- **N3**：[NIST/SEMATECH，§1.3.3.14 Histogram](https://www.itl.nist.gov/div898/handbook/eda/section3/histogra.htm)。

本課以所列來源的概念作繁體中文摘要，並加入自行編寫的情境、手算、程式及練習；不是原網站的逐段翻譯。標為「來源摘要」者為改寫，標為「自編」者為本教材教學設計；短引文另以引號及「中譯」標明。來源連結可供對照原文，查閱日期：2026-11-17。


### 短引文對照

> “The mean is the most common measure of the center.”
>
> 中譯：「平均數是最常用的中心位置度量。」
>
> 出處：OpenStax，§2.5（[原文](https://openstax.org/books/introductory-statistics-2e/pages/2-5-measures-of-the-center-of-the-data)）。此短句譯文由本教材提供；上方計算與產品例子為自編。
