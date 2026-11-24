# 課程大綱：統計基礎、深度學習與論文閱讀

## 對象與教學目標

適合商管及跨領域學生，需具 Python 基礎。以小型手算、CPU 實驗與論文閱讀，銜接 DLI 深度學習及 LLM 應用。每課採問題情境、先備知識、公式符號、手算、程式、DLI 對照、誤解與來源的順序。

## 學期導讀

先讀 [從統計到 AI 的模型設計](statistics-to-ai.md) 的閱讀路線，再到 [十課講義索引](units/README.md) 按先備知識選課。先釐清資料及切分，再建立模型，最後用任務指標與實驗證據解讀效果。各課先讀圖與手算，再到 Notebook 改動一個條件並解釋結果。

## 新版教學順序

下表使用新版授課順序。表中的單元編號用來排課，現有 Notebook 編號用來找檔案。部分主題已有講義導讀，是否已有獨立練習可從「現有教材與狀態」欄查看。

| 單元 | 主題 | 現有教材與狀態 | 論文連結 |
|---|---|---|---|
| 01 | 資料、樣本與分布 | Notebook 01；抽樣另參考 03 | 論文共通 |
| 02 | 平均、變異數與標準差 | Notebook 01 | PowerNorm |
| 03 | 機率、期望與條件機率 | [Notebook 02](../notebooks/02_probability_classification.ipynb)已有機率與條件機率，期望值與 Bernoulli 練習待補 | RAG、T5 |
| 04 | 張量與統計軸 | [文字統計講義](units/10-text-statistics.md)已有 token 表示形狀的例子，張量切片、沿軸計算與獨立練習待補 | PowerNorm、ViT |
| 05 | 標準化、均方與 RMS | [尺度講義](units/04-scaling-normalization.md)、Notebook 04 與背景導讀 | PowerNorm |
| 06 | BN、LN 與 PowerNorm | [背景導讀](statistics-to-ai.md)已有統計軸比較與 RMS 手算，同一張量上的 BN／LN 前向實驗待補，PowerNorm 更新與反向列進階閱讀 | PowerNorm |
| 07 | 迴歸殘差、機率輸出與損失 | [講義](units/05-statistical-models-loss.md)、Notebook 05 | T5、RAG |
| 08 | Mini-batch 與隨機梯度 | [損失講義](units/05-statistical-models-loss.md)已有單筆梯度手算，批次平均與更新比較待補 | Chinchilla |
| 09 | Dropout 與正則化 | [Notebook 06](../notebooks/06_overfitting_generalization.ipynb)已有正則化練習，Dropout 遮罩、縮放期望與訓練／推論差異待補 | 訓練基礎 |
| 10 | 啟動函數、資料切分與泛化 | [講義](units/06-overfitting-generalization.md)、Notebook 03、06 | T5、ViT、Whisper |
| 11 | 分類指標與決策 | Notebook 07 | Whisper、RAG；任務指標需延伸 |
| 12 | 不確定性與公平比較 | Notebook 08、09 | 所有實驗型論文 |
| 13 | 文字統計、資訊量與 Transformer 殘差連接 | [講義](units/10-text-statistics.md)、Notebook 10 | T5、LLM-Embedder |
| E1–E4 | 規模、弱監督、檢索與 RAG 評估 | 閱讀導引與練習設計已加入；Notebook 待新增 | 八篇論文整合 |

完整論文題名、原文、先備統計、練習及辨誤見 [論文閱讀導引](paper-reading-guide.md)。

## 待補教材的開發順序

| 優先順序 | 要補的內容 | 學生完成後能做什麼 |
|---|---|---|
| 1. 補齊機率與張量基礎 | 在現有機率課加入期望值、Bernoulli 與手算。在獨立的張量練習中對照影像批次和 token 表示，沿不同軸算平均、變異數或 RMS。 | 說清楚每一軸代表什麼、哪些數值參與統計，以及結果的形狀。 |
| 2. 接到模型內部的正規化 | 用小張量比較 BN 與 LN 的統計範圍和訓練／推論行為。以均方與 RMS 讀 PowerNorm 的設計，反向推導留到論文研讀。 | 用手算解釋同一輸入在兩種正規化中為什麼得到不同結果，並用 CPU Notebook 核對。 |
| 3. 接到訓練更新與正則化 | 延續迴歸與反向傳播例子，比較單筆、mini-batch、整批的梯度和更新頻率。用 Bernoulli 遮罩手算 Dropout，再用固定種子對照訓練和推論。 | 看懂批次大小改變了什麼，並解釋 Dropout 的縮放與期望如何相連。 |
| 4. 補足任務評估與選修 | 在分類指標之後補語音的 WER 與檢索的排序指標。保留 E1–E4 的論文閱讀設計，分別補可執行的選修練習。 | 先選符合任務的指標，再以小資料完成相應的模型比較或誤差分析。 |

新增的實作延續現有教材的格式。先做圖解和手算，再用可在 CPU 執行的 Notebook 核對結果，最後安排一個可修改條件的練習，並標明 [DLI](dli-mapping.md) 或 [論文閱讀](paper-reading-guide.md) 的對應位置。

## 時數建議

- 現有十份入門 Notebook：原規劃 20 小時，包含前七課 14 小時及進階三課 6 小時。
- 新增統計運算橋接：建議另排 6–8 小時；需依張量、BN／LN／PN、隨機梯度與 Dropout 實作完成程度調整。
- 論文研讀 E1–E4：選修 8 小時。其餘論文可穿插既有單元。
- 上述為教學估計，不含完整 DLI 訓練、模型下載、論文重現或大型預訓練。

## 作業與完成標準

1. 用手算解釋一個統計運算。
2. 在 Notebook 驗證並說明參數變動影響。
3. 對照一篇論文，區分作者結論、教學簡化與自身實驗。
4. 報告資料切分、指標定義、重複結果、限制與來源版本。
