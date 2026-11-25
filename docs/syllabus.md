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
| 03 | 機率、期望與條件機率 | [機率與分類](units/02-probability-classification.md) · [Notebook 02](../notebooks/02_probability_classification.ipynb)，含期望成本與 Bernoulli 抽樣 | RAG、T5 |
| 04 | 張量與統計軸 | [張量與統計軸](units/bridge-tensor-axes.md) · [Notebook](../notebooks/bridge_tensor_axes.ipynb)，沿不同軸平均、變異數、RMS 與填補遮罩 | PowerNorm、ViT |
| 05 | 標準化、均方與 RMS | [尺度講義](units/04-scaling-normalization.md)、Notebook 04 與背景導讀 | PowerNorm |
| 06 | BN、LN 與 PowerNorm | [模型內正規化：BN、LN 與 PowerNorm](units/bridge-normalization-layers.md) · [Notebook](../notebooks/bridge_normalization_layers.ipynb)，BN／LN 前向比較與均方平滑練習，PowerNorm 完整更新與反向見原論文 | PowerNorm |
| 07 | 迴歸殘差、機率輸出與損失 | [講義](units/05-statistical-models-loss.md)、Notebook 05 | T5、RAG |
| 08 | Mini-batch 與隨機梯度 | [Mini-batch 與隨機梯度](units/bridge-minibatch-gradients.md) · [Notebook](../notebooks/bridge_minibatch_gradients.ipynb)，批次梯度、更新次數與相同樣本預算比較 | Chinchilla |
| 09 | Dropout 與正則化 | [第 06 課講義](units/06-overfitting-generalization.md)與 [Notebook 06](../notebooks/06_overfitting_generalization.ipynb)已有 L2 懲罰、Bernoulli 遮罩、期望縮放與訓練／推論練習 | 訓練基礎 |
| 10 | 啟動函數、資料切分與泛化 | [講義](units/06-overfitting-generalization.md)、Notebook 03、06 | T5、ViT、Whisper |
| 11 | 分類指標與決策 | [分類模型評估](units/07-classification-evaluation.md) · [Notebook 07](../notebooks/07_classification_evaluation.ipynb)，含 WER、Recall@k 與 MRR | Whisper、RAG |
| 12 | 不確定性與公平比較 | Notebook 08、09 | 所有實驗型論文 |
| 13 | 文字統計、資訊量與 Transformer 殘差連接 | [講義](units/10-text-statistics.md) · Notebook 10 · [從零訓練 Transformer 實作](transformer-training-lab.md) | T5、LLM-Embedder |
| E1–E4 | 規模、弱監督、檢索與 RAG 評估 | [E1](units/e1-compute-scaling.md) · [E2](units/e2-weak-supervision-wer.md) · [E3](units/e3-retrieval-probability.md) · [E4](units/e4-rag-ablation.md)，附原文章節與表格對照、手算和 CPU Notebook | Chinchilla、Whisper、RAG、LLM-Embedder、RAG Survey |

整門課有八篇論文。PowerNorm、T5、ViT 搭配主課閱讀；Chinchilla、Whisper、RAG、LLM-Embedder 和 RAG Survey 放在 E1–E4 四個選修模組，其中 RAG 原論文同時用於 E3 與 E4。[八篇論文入口](../Papers.md) · [論文閱讀導引](paper-reading-guide.md) · [重要名詞解釋](paper-glossary.md)

## 補充單元的閱讀順序

1. 從期望值開始，再讀[張量與統計軸](units/bridge-tensor-axes.md)，先說清楚「哪個結果取期望」和「張量的哪些數值要一起算」。
2. 用 [BN／LN](units/bridge-normalization-layers.md) 追一筆資料如何依統計範圍得到不同輸出，再讀 [mini-batch](units/bridge-minibatch-gradients.md)，確認每批怎麼算梯度與更新。第 06 課的 Dropout 延用伯努利遮罩與期望。
3. E1 接在 mini-batch 後理解訓練配額。E2 接在任務評估後練習 WER。E3 先做檢索排序和機率融合，E4 再把檢索、答案、引用與延遲放在同一個實驗。

每個新增 Notebook 先以合成資料核對手算，再改一個條件看圖表如何變化。與 DLI 實作的關係可查 [DLI 對照](dli-mapping.md)。

## 時數建議

- 現有十份入門 Notebook：原規劃 20 小時，包含前七課 14 小時及進階三課 6 小時。
- 統計運算橋接：張量軸、BN／LN／PN、mini-batch 與 Dropout 建議另排 6–8 小時，依學生的 NumPy 與梯度基礎調整。
- 論文研讀 E1–E4：選修 8 小時。其餘論文可穿插既有單元。
- 上述為教學估計，不含完整 DLI 訓練、模型下載、論文重現或大型預訓練。

## 作業與完成標準

1. 用手算解釋一個統計運算。
2. 在 Notebook 驗證並說明參數變動影響。
3. 對照一篇論文，區分作者結論、教學簡化與自身實驗。
4. 報告資料切分、指標定義、重複結果、限制與來源版本。
