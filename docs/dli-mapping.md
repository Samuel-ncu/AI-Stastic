# DLI 章節與統計單元對照

原教材：[course_content/tutorials](https://github.com/NVDLI/fundamentals-of-deep-learning-zh/tree/main/course_content/tutorials)。以下為本專案建議的搭配順序。

| DLI Notebook | 統計補充 | 教學焦點 |
|---|---|---|
| 00_jupyterlab.ipynb | Python 與圖表暖身 | 陣列、資料表、圖形判讀 |
| 01_mnist.ipynb | 01、02、04、05 | 像素分布、尺度、分類、損失 |
| [02 手語影像分類](https://github.com/NVDLI/fundamentals-of-deep-learning-zh/blob/main/course_content/tutorials/02_asl.ipynb) | 03、05、06、[09](units/09-experimental-design.md) | 建立手語分類基準，記錄資料切分、初始設定與驗證指標，供後面比較 CNN 或資料增強。 |
| 03_asl_cnn.ipynb | 06、08、[CNN 補充講義](cnn-convolution-feature-maps.md) | 先追蹤各層輸入與輸出通道，再比較模型複雜度與評估條件。 |
| [04a 手語資料增強](https://github.com/NVDLI/fundamentals-of-deep-learning-zh/blob/main/course_content/tutorials/04a_asl_augmentation.ipynb) | 03、06、[09](units/09-experimental-design.md) | 用同一切分、模型及訓練預算，對照有無增強。用相同的一組訓練種子重跑兩組，比較驗證指標。 |
| [04b 手語推論](https://github.com/NVDLI/fundamentals-of-deep-learning-zh/blob/main/course_content/tutorials/04b_asl_predictions.ipynb) | 02、07 | 用 argmax 把 logits 轉為字母判定，再逐類讀混淆矩陣。 |
| [05a 使用預訓練模型開狗門](https://github.com/NVDLI/fundamentals-of-deep-learning-zh/blob/main/course_content/tutorials/05a_doggy_door.ipynb) | 07、08 | 先選最高分 ImageNet 類別，再依類別索引判斷是否開門。 |
| [05b 寵物門與遷移學習](https://github.com/NVDLI/fundamentals-of-deep-learning-zh/blob/main/course_content/tutorials/05b_corgi_door.ipynb) | 05、07、08、[09](units/09-experimental-design.md) | 一個 logit 搭配 BCE，Penny 為標籤 0，開門用 logit 小於 0。另記錄凍結與微調的評估條件。 |
| [06 自然語言處理](https://github.com/NVDLI/fundamentals-of-deep-learning-zh/blob/main/course_content/tutorials/06_nlp.ipynb) | [10](units/10-text-statistics.md) | 把 DLI 的 BERT 分詞、遮罩詞預測和問答，接到 token 表示、候選詞機率與注意力。 |

**第 09 課用來設計比較。** 例如做完 04a，可以用同一組種子比較有無資料增強的驗證結果。先固定評估方式與選擇規則，再用保留的測試集做最後評估。

**第 10 課用來讀懂模型。** DLI 06 用 BERT 根據上下文補回被遮住的 token，也做從文本擷取答案的練習。第 10 課先用詞袋和熵建立直覺，再補上嵌入向量、Q、K、V 與殘差連接；GPT 的下一個 token 預測是延伸對照。

讀 DLI 02 手語分類時，可用 [張量與統計軸](units/bridge-tensor-axes.md) 辨認一批影像的各維度，再用 [Mini-batch 與隨機梯度](units/bridge-minibatch-gradients.md) 追蹤每批損失與更新。閱讀 DLI 06 的 BERT 時，可搭配 [模型內正規化](units/bridge-normalization-layers.md) 釐清 token 的特徵在哪一軸做 LayerNorm。

單元編號與原教材章節編號各自獨立。原教材更新後需重新檢查對照。
