# 統計教學 Notebook

每份均包含概念說明、可執行 Python 範例、解讀及學生練習。使用 CPU，資料為合成或 scikit-learn 內建資料，無須下載 DLI 資料集。

在 GitHub 點選檔案可閱讀；點選 Colab 可執行。安裝儲存格需要網路。Python 執行驗證狀態請查看儲存庫 Actions。

| 單元 | 教材 | Colab |
|---|---|---|
| 資料與描述統計 | [Notebook](01_descriptive_statistics.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/01_descriptive_statistics.ipynb) |
| 機率與分類 | [Notebook](02_probability_classification.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/02_probability_classification.ipynb) |
| 抽樣與資料切分 | [Notebook](03_sampling_data_splits.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/03_sampling_data_splits.ipynb) |
| 尺度與標準化 | [Notebook](04_scaling_normalization.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/04_scaling_normalization.ipynb) |
| 統計模型與損失函數 | [Notebook](05_statistical_models_loss.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/05_statistical_models_loss.ipynb) |
| 過擬合與泛化 | [Notebook](06_overfitting_generalization.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/06_overfitting_generalization.ipynb) |
| 分類模型評估 | [Notebook](07_classification_evaluation.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/07_classification_evaluation.ipynb) |
| 不確定性與模型比較 | [Notebook](08_uncertainty_comparison.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/08_uncertainty_comparison.ipynb) |
| 實驗設計與重現 | [Notebook](09_experimental_design.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/09_experimental_design.ipynb) |
| 文字資料的統計 | [Notebook](10_text_statistics.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/10_text_statistics.ipynb) |

## 張量、訓練與論文選修 Notebook

左欄沿用[課綱的授課順序](../docs/syllabus.md)。先讀講義的手算，再用 Notebook 的圖表核對。

| 單元 | 教材 | Colab |
|---|---|---|
| 04 張量與統計軸 | [講義](../docs/units/bridge-tensor-axes.md) · [Notebook](bridge_tensor_axes.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/bridge_tensor_axes.ipynb) |
| 06 模型內正規化：BN、LN 與 PowerNorm | [講義](../docs/units/bridge-normalization-layers.md) · [Notebook](bridge_normalization_layers.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/bridge_normalization_layers.ipynb) |
| 08 Mini-batch 與隨機梯度 | [講義](../docs/units/bridge-minibatch-gradients.md) · [Notebook](bridge_minibatch_gradients.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/bridge_minibatch_gradients.ipynb) |
| E1 模型規模與計算配額 | [講義](../docs/units/e1-compute-scaling.md) · [Notebook](e1_compute_scaling.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/e1_compute_scaling.ipynb) |
| E2 弱監督與語音評估 | [講義](../docs/units/e2-weak-supervision-wer.md) · [Notebook](e2_weak_supervision_wer.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/e2_weak_supervision_wer.ipynb) |
| E3 檢索排序與機率融合 | [講義](../docs/units/e3-retrieval-probability.md) · [Notebook](e3_retrieval_probability.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/e3_retrieval_probability.ipynb) |
| E4 RAG 消融與來源評估 | [講義](../docs/units/e4-rag-ablation.md) · [Notebook](e4_rag_ablation.ipynb) | [開啟](https://colab.research.google.com/github/Samuel-ncu/AI-Stastic/blob/main/notebooks/e4_rag_ablation.ipynb) |

## 本機執行

```bash
pip install -r requirements.txt
jupyter lab
```

逐格執行；學生練習需自行修改程式。這些是統計入門實驗，未包含完整 DLI 神經網路訓練或三個核心專題的全部實作。
