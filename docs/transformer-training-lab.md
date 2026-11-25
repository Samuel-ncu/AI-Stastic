# 補充實作｜用 Tiny Shakespeare 從零訓練一個 Transformer

第 05 課用交叉熵評估模型給真實類別多少機率，第 10 課用 Q、K、V 算注意力。這篇用同一個訓練迴圈把兩者接起來。模型從隨機初始化開始，讀一段文字，在每個位置猜下一個字元，再用猜錯的程度更新權重。

[第 05 課：統計模型與損失](units/05-statistical-models-loss.md) · [第 10 課：文字資料的統計](units/10-text-statistics.md)

## 1. 先選資料集和任務

第一次實作可用 **Tiny Shakespeare ＋ Karpathy 的《Let’s build GPT》**。一個純文字檔就能練習分割資料、建立字元 ID、組成 mini-batch，並追蹤損失與生成結果。這個模型使用 GPT 式的 decoder-only 架構。

| 想練什麼？ | 可取得的資料 | 搭配教材 |
|---|---|---|
| 從零訓練小型 GPT，預測下一個字元 | [Tiny Shakespeare 原始文字檔](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) | [Karpathy 影片](https://www.youtube.com/watch?v=kCc8FmEb1nY) · [PyTorch 原始碼](https://github.com/karpathy/ng-video-lecture) · [Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) |
| 生成短篇故事 | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 有訓練與驗證資料。課堂可從小子集開始。 | [TinyStories 論文](https://arxiv.org/abs/2305.07759)，可用來討論資料設計如何影響小模型。 |
| 看原版 Transformer 的編碼器、解碼器和交叉注意力 | [TED 葡萄牙文到英文翻譯資料](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate) | [TensorFlow 官方翻譯教學](https://www.tensorflow.org/text/tutorials/transformer)，內附 Colab。 |
| 微調預訓練的 BERT 做二元分類 | [GLUE／MRPC](https://huggingface.co/datasets/nyu-mll/glue)，每筆包含兩句話和「意思是否相同」的標籤。 | [Hugging Face 的完整 PyTorch 訓練迴圈](https://huggingface.co/learn/llm-course/chapter3/4) |

## 2. 一段文字怎麼變成四個答案？

第一次讀程式時，從 *input.txt* 的開頭取出 *First*。以字元作為 token，輸入從第一個字元讀到第四個，答案則從第二個讀到第五個。實際送進模型的是這些字元各自對應的整數 ID。

| 位置 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 輸入 $X$ | F | i | r | s |
| 下一個字元 $Y$ | i | r | s | t |

例如第 3 個位置能讀到 *Fir*，要猜下一個字元 *s*。雖然 *s* 也在這批輸入的第 4 格，**因果遮罩（causal mask）會擋住右邊的位置**，所以第 3 格的注意力看不到它。整批資料能同時計算四個位置的損失，仍遵守每個位置只能使用當下與前文的規則。

上課時先記錄資料切分方式。Karpathy 的示範用全文字元集合建立 ID 對照，前 90% 文字用來訓練，後 10% 用來驗證，再各自在自己的區段內抽視窗。若改用多篇文章或故事，先以整篇為單位切分，再從各集合取視窗。如果還要從語料學 BPE 等 tokenizer，只用訓練集估計切詞規則，再用同一套規則處理留出的資料。[第 03 課：抽樣與資料切分](units/03-sampling-data-splits.md) · [原始程式](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)

## 3. 追一次前向運算、損失與權重更新

1. **字元 ID 變成向量。** 把字元嵌入和位置嵌入相加，讓模型有字元內容與位置資訊。
2. **Q、K、V 建立上下文。** 每個注意力頭用目前的向量算出 Q、K、V。Q 和 K 決定這個位置要讀前文的哪些位置，再用注意力權重加總對應的 V。
3. **輸出 logits，對照真實字元。** 每個位置輸出一組詞彙表分數。假設第 3 格把 25% 機率給正確字元 *s*，該格交叉熵是 $-\ln(0.25)\approx1.386$。若下一次給了 50%，損失降到 $-\ln(0.5)\approx0.693$。把批內各位置的損失平均，就是這次更新用的損失。
4. **反向傳播，再由最佳化器更新。** 原始程式先清除舊梯度，再用 *loss.backward()* 算本次梯度，最後用 *optimizer.step()* 更新參數。字元嵌入、注意力投影和輸出層等可訓練參數，都能透過同一個損失得到梯度。下一批會用更新後的參數重新算分數。[原始程式](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) · [Mini-batch 與隨機梯度](units/bridge-minibatch-gradients.md)

這裡有兩個 Softmax。**注意力的 Softmax 沿可讀取的文字位置分配權重**，回答「要參考哪裡？」。**預測機率的 Softmax 沿詞彙表分配機率**，回答「接下來可能是哪個字元？」。原始程式的訓練損失直接接收 logits，生成文字時才顯式把 logits 換成機率來抽樣。[第 10 課：Q、K、V 與輸出熵](units/10-text-statistics.md) · [原始程式](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)

## 4. 先跑一次小模型，再看它學到了什麼

開啟 Karpathy 的 Colab，或把 *gpt.py* 與 *input.txt* 放在同一工作目錄後執行程式。初次練習可先用下表的縮小設定，觀察一次完整流程。這些是課堂起點，訓練速度和結果取決於執行環境與設定。[開啟 Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) · [開啟 gpt.py](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)

| 設定 | 建議起點 | 在這個實作中的意思 |
|---|---:|---|
| *batch_size* | 16 | 每次更新取 16 段。 |
| *block_size* | 32 | 每段有 32 個輸入字元。 |
| *n_embd*、*n_head*、*n_layer* | 64、4、2 | 64 維向量、4 個注意力頭、2 個 Transformer block。 |
| *max_iters*、*eval_interval*、*eval_iters* | 400、100、20 | 執行 400 次參數更新，每隔 100 步取樣評估。每次各抽 20 批估計訓練與驗證損失。 |

課堂紀錄可以包含三項：

1. **印一批 $X$ 和 $Y$。** 解碼回文字，確認每列的答案確實往後移一個字元。上述設定每批有 $16\times 32=512$ 個預測位置。
2. **畫出訓練和驗證損失。** 用步數當橫軸，平均交叉熵當縱軸。訓練損失持續下降、驗證損失開始反升時，可能開始過擬合。原程式會隨機抽驗證批次，曲線也會有抽樣波動。[第 06 課：過擬合與泛化](units/06-overfitting-generalization.md)
3. **用同一段開頭比較訓練前後的生成。** 例如都用 *First Citizen:* 當提示，在訓練迴圈前後各產生同樣長度的文字。固定生成設定，觀察空白、標點和字詞片段的變化，再和驗證損失一起解讀。單次生成結果也會受抽樣影響。

可在原始程式建立模型後加入下方函數，並在訓練迴圈前、後各呼叫一次 `show_generation()`。這會讓兩次比較使用同一段開頭與相同的輸出長度。

```python
def show_generation():
    prompt = "First Citizen:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=120)
    model.train()
    print(decode(generated[0].tolist()))
```

### 如何連回 NVIDIA DLI？

DLI 06 的 BERT 練習是填入句中被遮住的 token，這篇的 GPT 則用左側前文猜下一個字元。學生可拿同一句話比較「哪些位置看得到哪些字」和「答案放在哪一格」，再回來看兩個任務如何用輸出分數選 token。[第 10 課的 DLI 說明](units/10-text-statistics.md) · [DLI 對照表](dli-mapping.md)
