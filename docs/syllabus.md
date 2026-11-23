# 課程大綱：統計基礎、深度學習與論文閱讀

## 對象與教學目標

適合商管及跨領域學生，需具 Python 基礎。以小型手算、CPU 實驗與論文閱讀，銜接 DLI 深度學習及 LLM 應用。每課採問題情境、先備知識、公式符號、手算、程式、DLI 對照、誤解與來源的順序。

## 學期導讀

先讀 [從統計到 AI 的模型設計](statistics-to-ai.md)，再以既有 02、05、06、10 課的講義與 Notebook 練習。導讀把迴歸、殘差、啟動與損失函數、正規化和 Transformer 串成一條閱讀路線。

## 新版教學順序

這是整合後的課程編號，與既有 Notebook 檔名不同。既有十份 Notebook 保留原路徑，避免破壞連結；新單元尚待開發，不宣稱已完成。

| 單元 | 主題 | 現有教材／狀態 | 論文連結 |
|---|---|---|---|
| 01 | 資料、樣本與分布 | Notebook 01；抽樣另參考 03 | 論文共通 |
| 02 | 平均、變異數與標準差 | Notebook 01 | PowerNorm |
| 03 | 機率、期望與條件機率 | Notebook 02；期望值需擴充 | RAG、T5 |
| 04 | 張量與統計軸 | 待新增 | PowerNorm、ViT |
| 05 | 標準化、均方與 RMS | [尺度講義](units/04-scaling-normalization.md)、Notebook 04 與背景導讀 | PowerNorm |
| 06 | BN、LN 與 PowerNorm | 待新增；前向直覺必修，PN 反向進階 | PowerNorm |
| 07 | 迴歸殘差、機率輸出與損失 | [講義](units/05-statistical-models-loss.md)、Notebook 05 | T5、RAG |
| 08 | Mini-batch 與隨機梯度 | 待新增 | Chinchilla |
| 09 | Dropout 與正則化 | Notebook 06 涵蓋部分；Bernoulli 與期望待新增 | 訓練基礎 |
| 10 | 啟動函數、資料切分與泛化 | [講義](units/06-overfitting-generalization.md)、Notebook 03、06 | T5、ViT、Whisper |
| 11 | 分類指標與決策 | Notebook 07 | Whisper、RAG；任務指標需延伸 |
| 12 | 不確定性與公平比較 | Notebook 08、09 | 所有實驗型論文 |
| 13 | 文字統計、資訊量與 Transformer 殘差連接 | [講義](units/10-text-statistics.md)、Notebook 10 | T5、LLM-Embedder |
| E1–E4 | 規模、弱監督、檢索與 RAG 評估 | 閱讀導引與練習設計已加入；Notebook 待新增 | 八篇論文整合 |

完整論文題名、原文、先備統計、練習及辨誤見 [論文閱讀導引](paper-reading-guide.md)。

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
