# 論文閱讀與統計背景對照

## 課程定位

以「統計基礎 → 深度學習運算 → 研究論文與 LLM 應用」銜接 DLI Fundamentals of Deep Learning 與 Rapid Application Development with LLMs。

論文清單來自 Samuel 的 [Papers.md](https://github.com/Samuel-ncu/DLI-Rapid_Application_Development_with_Large_Language_Models/blob/main/Papers.md)。本頁以原論文摘要／出版頁核對題名、方法概述與課程定位，提供中文摘要及自編練習，不是論文全文翻譯或完整重現。涉及方法細節與實驗數據時，仍需閱讀原文指定版本。

課程單元編號對應 [整合課程大綱](syllabus.md) 的「新版教學順序」，不是目前 Notebook 檔名。已完成 Notebook 與待補教材在課綱中分開標示。

## 論文與統計背景總表

| 編號 | 論文 | 對應單元 | 需要的統計背景 |
|---|---|---|---|
| P01 | [PowerNorm: Rethinking Batch Normalization in Transformers](https://proceedings.mlr.press/v119/shen20e.html) | 06 BN／LN／PN | 二次動差、RMS、移動平均、批次統計波動 |
| P02 | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer（T5）](https://www.jmlr.org/papers/v21/20-074.html) | 07 機率與損失、10 泛化、13 文字統計 | 條件機率、序列交叉熵、遷移學習、控制變因 |
| P03 | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ViT）](https://arxiv.org/abs/2010.11929v2) | 04 張量與統計軸、10 泛化 | 樣本量、歸納偏置、資料分布、遷移評估 |
| P04 | [Training Compute-Optimal Large Language Models（Chinchilla）](https://arxiv.org/abs/2203.15556) | 08 批次與訓練、12 模型比較；延伸 E1 | 冪律、對數座標、迴歸擬合、受限最佳化、外推限制 |
| P05 | [Robust Speech Recognition via Large-Scale Weak Supervision（Whisper）](https://arxiv.org/abs/2212.04356) | 10 泛化、11 評估、12 不確定性；延伸 E2 | 弱監督、標籤雜訊、分布差異、分層評估、WER |
| P06 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401v4) | 03 條件機率、07 損失、11 評估；延伸 E3 | 潛在變數、邊際化、條件機率、檢索與生成的分開評估 |
| P07 | [Retrieve Anything To Augment Large Language Models（LLM-Embedder）](https://arxiv.org/abs/2310.07554v2) | 13 向量表示；延伸 E3、E4 | 相似度、排名、負樣本抽樣、蒸餾、多任務加權 |
| P08 | [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997v5) | 11 評估、12 實驗設計；延伸 E3、E4 | 評估構念、指標分解、消融實驗、樣本偏誤 |

## 建議閱讀順序

- CNN 與正規化：先完成平均、變異數、RMS、統計軸，再讀 P01；連到原課程 BatchNorm。
- 遷移與表示：P02 → P03，再用 P04 討論資料、模型大小與計算限制。
- 跨領域泛化：P05；不以語音辨識分數直接替代文字或影像任務的指標。
- RAG：P06 → P07 → P08；分清檢索器表現、生成品質及來源支持度。

## P01 PowerNorm: Rethinking Batch Normalization in Transformers

**年份／版本：**2020，ICML。 [原始來源](https://proceedings.mlr.press/v119/shen20e.html)。

**原文概念中文摘要：**研究 Transformer 中跨批次統計波動，提出放寬零均值要求、使用 running quadratic mean，並搭配近似反向傳播的 PowerNorm。

**先備統計：**二次動差、RMS、移動平均、批次統計波動。

**建議插入位置：**06 BN／LN／PN。

**本教材自編練習：**用 [2,4] 手算平均、變異數與 RMS；說明為何中心化後的標準差與未中心化的 RMS 不相同。

**閱讀限制與辨誤：**PN 不是任意選 p 的 p 次方正規化，也不等於只把 LN 的減平均刪掉。完整演算法須閱讀論文方法與實作；本導引不宣稱已重現。

## P02 Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer（T5）

**年份／版本：**2019 預印本／2020 JMLR。 [原始來源](https://www.jmlr.org/papers/v21/20-074.html)。

**原文概念中文摘要：**將多種文字任務統一為 text-to-text，系統比較預訓練目標、架構、資料與遷移方式。

**先備統計：**條件機率、序列交叉熵、遷移學習、控制變因。

**建議插入位置：**07 機率與損失、10 泛化、13 文字統計。

**本教材自編練習：**把分類與摘要各改寫成輸入／輸出文字；設計一個只改預訓練目標的對照表。

**閱讀限制與辨誤：**不同資料與調參預算的分數不能直接歸因於架構。教學不沿用未逐項核對的跨模型 GLUE 排名。

## P03 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ViT）

**年份／版本：**2020 預印本／2021 ICLR。 [原始來源](https://arxiv.org/abs/2010.11929v2)。

**原文概念中文摘要：**把影像 patch 當作序列輸入 Transformer，研究大規模預訓練後的影像辨識遷移表現。

**先備統計：**樣本量、歸納偏置、資料分布、遷移評估。

**建議插入位置：**04 張量與統計軸、10 泛化。

**本教材自編練習：**自編例：224×224 影像、16×16 patch，計算 14×14=196 個 patch token，另區分類別 token；列出公平比較 CNN 與 ViT 要固定的條件。

**閱讀限制與辨誤：**16×16 是特定 patch 設定，並非所有 ViT 固定使用此大小。預訓練資料、解析度及微調設定都影響比較；不要宣稱 CNN 完全無法建模全域資訊。

## P04 Training Compute-Optimal Large Language Models（Chinchilla）

**年份／版本：**2022。 [原始來源](https://arxiv.org/abs/2203.15556)。

**原文概念中文摘要：**研究固定計算預算下模型參數量與訓練 token 數的配置；其實驗結論支持兩者近似等比例擴增。

**先備統計：**冪律、對數座標、迴歸擬合、受限最佳化、外推限制。

**建議插入位置：**08 批次與訓練、12 模型比較；延伸 E1。

**本教材自編練習：**自編例：在 C≈6ND 假設下，若 N、D 各加倍，計算量約增為四倍；說明此近似沒有納入哪些部署成本。

**閱讀限制與辨誤：**不可沿用原筆記 N∝C^0.73、D∝C^1.27 作為 Chinchilla 結論。近似均衡縮放不是任何資料、架構、預算下都固定成立的自然定律。

## P05 Robust Speech Recognition via Large-Scale Weak Supervision（Whisper）

**年份／版本：**2022。 [原始來源](https://arxiv.org/abs/2212.04356)。

**原文概念中文摘要：**以大規模多語、多任務語音與轉錄資料研究弱監督語音辨識及零樣本遷移。

**先備統計：**弱監督、標籤雜訊、分布差異、分層評估、WER。

**建議插入位置：**10 泛化、11 評估、12 不確定性；延伸 E2。

**本教材自編練習：**自編 WER 練習：參考句 10 詞，替換 1、刪除 1、插入 1，WER=(1+1+1)/10=30%；另按口音／語言分組報告。

**閱讀限制與辨誤：**WER=(S+D+I)/參考詞數，可因插入很多詞而超過 100%。比較需固定資料切分、文字正規化及解碼設定，不沿用原筆記未核對的模型 WER 表。

## P06 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**年份／版本：**2020，NeurIPS。 [原始來源](https://arxiv.org/abs/2005.11401v4)。

**原文概念中文摘要：**結合參數化 seq2seq 模型與非參數文件索引，提出序列層級及 token 層級兩種檢索增強生成建模。

**先備統計：**潛在變數、邊際化、條件機率、檢索與生成的分開評估。

**建議插入位置：**03 條件機率、07 損失、11 評估；延伸 E3。

**本教材自編練習：**用兩份文件的檢索權重與條件生成機率，手算一次加權總和；再分別記錄檢索成功與回答正確的比例。

**閱讀限制與辨誤：**RAG-Sequence 對整段序列以同一文件條件做混合，RAG-Token 可逐 token 對文件做混合；這不等於必定每產生一個 token 就重新呼叫檢索器。

## P07 Retrieve Anything To Augment Large Language Models（LLM-Embedder）

**年份／版本：**2023，本文固定閱讀 arXiv v2。 [原始來源](https://arxiv.org/abs/2310.07554v2)。

**原文概念中文摘要：**以統一 embedding 模型支援多種 LLM 檢索增強需求，研究 LLM 回饋、蒸餾、多任務訓練與同質批內負樣本。

**先備統計：**相似度、排名、負樣本抽樣、蒸餾、多任務加權。

**建議插入位置：**13 向量表示；延伸 E3、E4。

**本教材自編練習：**同一查詢列出候選文件與相關標籤，計算 Recall@k、MRR，並討論負樣本選擇如何影響訓練難度。

**閱讀限制與辨誤：**不把 Anything 直接解讀為已證明通用的圖片／表格／程式碼多模態索引器。arXiv 最新 v3 標題已改為 A Multi-Task Embedder For Retrieval Augmented LLMs；為對應原清單，本課固定 v2。

## P08 Retrieval-Augmented Generation for Large Language Models: A Survey

**年份／版本：**2023 初稿／2024 v5。 [原始來源](https://arxiv.org/abs/2312.10997v5)。

**原文概念中文摘要：**整理 Naive、Advanced、Modular RAG 及檢索、生成、增強環節與評估框架。

**先備統計：**評估構念、指標分解、消融實驗、樣本偏誤。

**建議插入位置：**11 評估、12 實驗設計；延伸 E3、E4。

**本教材自編練習：**畫出自己的 RAG 系統，為檢索、生成、來源支持度、延遲各設一個量測方式；設計移除 reranker 的消融實驗。

**閱讀限制與辨誤：**Survey 的分類不是所有系統唯一標準；自動評分或 LLM judge 分數需有定義、抽樣與人工核對。

## 自編公式橋接：從基本統計到論文

### PowerNorm 前置概念

$$E[X^2]=\operatorname{Var}(X)+(E[X])^2$$

對 [2,4]，平均數為 3，以除以 n 定義的變異數為 1，均方值為 10，RMS 為 $\sqrt{10}$。標準差與 RMS 不同。這是前置數值練習，**不是 PowerNorm 完整演算法**；PN 還包含移動統計量與近似反向傳播。

### RAG 前置概念：對文件做邊際化

$$p(y\mid x)\approx\sum_{z\in \mathrm{top}\text{-}k}p(z\mid x)\,p(y\mid x,z)$$

把 x 視為問題、z 視為候選文件、y 視為回答。這是有限候選文件下的教學近似，實際公式與正規化方式應對照論文。自編數字：候選權重 [0.7,0.3]，同一離散回答 y 的條件機率 [0.8,0.2]，加權總和為 0.62。此計算不自動等於人類判定回答事實正確的機率。

### 檢索評估前置概念

Recall@k 是前 k 筆取回的相關項目數除以所有已標註相關項目數；MRR 是各查詢第一個相關結果之排名倒數的平均。本課對找不到相關結果的查詢記 0。部分問答文獻把 top-k 是否包含任一答案文件的命中率稱為 recall，報告需明確定義。

例如兩個查詢的首個相關結果在第 1、4 名，MRR=(1+1/4)/2=0.625。這是自編例，不是任何論文的實測結果。

## 延伸教學模組（設計完成，尚無新增 Notebook）

| 模組 | 內容 | 建議時間 |
|---|---|---:|
| E1 規模與計算 | 冪律、log-log 圖、參數／token 配置與外推限制 | 2 小時 |
| E2 弱監督與語音評估 | 標籤雜訊、WER、語言／口音分層與公平比較 | 2 小時 |
| E3 檢索與機率融合 | 相似度、邊際化、Recall@k、MRR、回答品質 | 2 小時 |
| E4 RAG 實驗設計 | 消融、檢索與生成分開評估、延遲與來源核對 | 2 小時 |

## 論文作業繳交規格

每組選一篇，提交「研究問題、先備統計、符號解釋、手算例子、實驗設計、結論範圍、原始來源與版本」。區分作者報告的結果與學生自行重現的結果。引用數據需附表／圖編號與設定；不同資料、解碼策略及預算的分數不直接放在同一排行榜。

查閱日期：2026-11-17。
