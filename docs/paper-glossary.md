# 八篇論文重要名詞解釋

讀原文時可以依 P01–P08 找名詞。每一列先說明概念，再用課程例子交代它在公式、架構或結果表中的用途。例子的數字用來練習讀法。

[八篇原論文與教材入口](../Papers.md) · [逐篇閱讀導引](paper-reading-guide.md)

## P01 PowerNorm：要沿哪一軸算統計量？

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| BN（Batch Normalization，批次正規化） | 同一特徵或通道用一組資料的平均與變異數來調整尺度。 | CNN 的 BatchNorm2d 常對每個通道跨批次及空間位置計算。 |
| LN（Layer Normalization，層正規化） | 用單筆輸入指定維度內的數值算平均與變異數。 | Transformer 常對每個 token 的特徵維度計算。 |
| 二次動差、RMS（均方根） | 二次動差是 $E[X^2]$，RMS 是 $\sqrt{E[X^2]}$。 | 用數值 2、4 等權計算，均方是 10，RMS 是 √10；中心化變異數是 1。 |
| Running quadratic mean（跨批次累積的均方統計） | 新批次到來時更新已累積的均方，用來減少單一批次波動的影響。 | 讀算法時分別追各批的均方、累積值，以及前向分母用了哪一個。 |
| PN（Power Normalization） | PowerNorm 放寬零均值要求，搭配跨批次的均方統計與近似反向傳播。 | 留意論文怎麼更新統計量和傳梯度。 |

[PowerNorm 原論文](https://proceedings.mlr.press/v119/shen20e.html) · [正規化講義](units/bridge-normalization-layers.md)

## P02 T5：怎麼把任務寫成文字輸入和輸出？

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| Text-to-text（文字到文字） | 把分類、翻譯、摘要等任務統一寫成輸入文字、輸出文字。 | 情緒分類可以輸入句子與任務提示，輸出「正面」。 |
| 預訓練與微調（pre-training、fine-tuning） | 先用大量資料學一般表示，再用目標任務資料調整權重。 | 讀表時分清預訓練目標、下游資料與可更新的參數。 |
| Encoder–decoder（編碼器與解碼器） | 編碼器依輸入形成表示，解碼器使用該表示與已生成前文，逐步預測輸出。 | 摘要時，文章送入編碼器，摘要 token 由解碼器依序產生。 |
| Span corruption（片段破壞） | 把連續的文字片段遮掉，訓練模型重建被移除的內容。 | T5 把每段缺失內容以一個標記替代。 |
| Sentinel token（片段標記） | 為不同缺失片段分配不同的特殊 token，讓輸出能對回原位置。 | 兩處刪除可用不同標記，解碼器輸出時用相同標記標出各段。 |
| NSP（Next Sentence Prediction，下一句預測） | 判斷兩句文字是否在原始資料中相鄰，是原筆記提到的 BERT 預訓練任務。 | 讀 T5 時另辨認其片段重建目標。 |

[T5 原論文](https://www.jmlr.org/papers/v21/20-074.html) · [統計模型與損失](units/05-statistical-models-loss.md)

## P03 ViT：影像怎麼變成序列？

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| Patch（影像區塊） | 把影像切成小塊，讓各塊依位置排成序列。 | 224×224 影像切成 16×16 區塊，得到 196 塊。 |
| Patch embedding（區塊嵌入） | 將每個區塊攤平，再線性投影到模型的特徵維度。 | 把區塊數和每個區塊的表示維度分開記。 |
| Class token（分類 token） | 額外加入一個可學的向量，最後取它的表示做分類。 | 196 個區塊加上分類 token，送入編碼器的長度是 197。 |
| Position embedding（位置嵌入） | 為每個序列位置加入位置資訊。 | 兩個像素內容相同的區塊，仍可來自不同位置。 |
| Inductive bias（歸納偏置） | 模型在看到資料前，架構已給某些模式優先性。 | 例如 CNN 的局部卷積假設鄰近像素特別相關。 |

[ViT 原論文](https://arxiv.org/abs/2010.11929v2) · [張量與統計軸](units/bridge-tensor-axes.md)

## P04 Chinchilla：固定算力怎麼分配？

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| 參數量 N、訓練 token 數 D | N 是模型可學的參數數量；D 是訓練期間處理的 token 數。 | 同一文字被訓練多輪時，要分清語料中獨立的文字量與訓練中實際處理的 token 數。 |
| FLOPs、FLOP/s | FLOPs 表示一項工作需要多少次浮點運算，FLOP/s 表示每秒可做多少次。 | 讀 Chinchilla 的訓練預算時用前者；估計設備速度時看後者。 |
| IsoFLOP（等算力） | 固定訓練算力，改變模型大小與訓練 token 配置。 | 在同一條等算力曲線找損失最低的一點。 |
| Scaling law（縮放定律） | 用已觀察到的實驗，擬合模型大小、資料量、算力和損失的關係。 | 先讀清楚橫縱軸與觀測範圍，超出範圍時屬於外推。 |

[Chinchilla 原論文](https://arxiv.org/abs/2203.15556) · [E1 講義](units/e1-compute-scaling.md)

## P05 Whisper：怎麼解讀語音辨識表現？

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| ASR（Automatic Speech Recognition，自動語音辨識） | 把語音轉成文字。 | 轉錄保留原語言；Whisper 論文的語音翻譯任務將非英語音訊翻成英文。 |
| Weak supervision（弱監督） | 用大量容易取得、品質有差異的標註來訓練。 | 網頁影片字幕可能對到實際語音，也可能漏字或只寫大意。 |
| Zero-shot（零樣本） | 這裡指將訓練好的模型直接應用於評估資料集，無針對該資料集再做微調。 | Whisper 論文看這種設定下，模型換到不同語域的表現。 |
| WER（Word Error Rate，字詞錯誤率） | 最少替換、刪除、插入次數之和，除以參考詞數 $N$，即 $(S+D+I)/N$。 | 參考有 10 詞，錯 3 次，WER 為 30%。 |
| RER（Relative Error Reduction，相對錯誤減少） | 把 WER 的下降幅度除以原本的 WER。 | WER 從 20% 降到 15%，少 5 個百分點，相對錯誤減少 25%。 |
| CER（Character Error Rate，字元錯誤率） | 以字元做編輯距離，再除以參考字元數。 | 評估中文時可報 CER，並交代標點、繁簡及正規化設定。 |

[Whisper 原論文](https://arxiv.org/abs/2212.04356) · [E2 講義](units/e2-weak-supervision-wer.md)

## P06 RAG 原論文：文件怎麼影響答案機率？

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| RAG（Retrieval-Augmented Generation，檢索增強生成） | 先找相關內容，再把內容提供給生成模型來回答。 | 例如先查產品手冊，再回答保固多久。 |
| Parametric／non-parametric memory（參數式／非參數式記憶） | 參數式知識承載於學到的模型權重，非參數式記憶可由外部文件索引提供。 | Lewis 的 RAG 同時使用生成模型與外部文章片段。 |
| DPR（Dense Passage Retrieval，稠密片段檢索） | 用兩個編碼器各自把查詢和片段轉成向量，再用內積找相符的片段。 | 查詢與片段可以先分別編碼，文件向量預先入索引。 |
| MIPS（Maximum Inner Product Search，最大內積搜尋） | 找出與查詢向量內積最大的候選。 | 兩邊都化成單位向量時，內積值等於餘弦相似度。 |
| Latent variable、marginalization（潛在變數、邊際化） | 訓練只有問答時，實際依據的文件可視為潛在變數，對候選文件的貢獻加權求和就是邊際化。 | 兩份文件權重 0.6／0.4，答案條件機率 0.2／0.9，合起來是 0.48。 |
| RAG-Sequence／RAG-Token | Sequence 先算每份文件下的整段機率，再混合文件。Token 先為每個位置混合文件條件，再乘上各位置的機率。 | 對照 E3 的兩 token 例，先圈出在哪一步對文件求和。 |

[Lewis 等人的 RAG 原論文](https://arxiv.org/abs/2005.11401v4) · [E3 講義](units/e3-retrieval-probability.md)

## P07 LLM-Embedder：檢索器怎麼學到排序？

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| Embedding（嵌入向量） | 把文字編碼為數值向量，後續可用向量做排序或比較。 | 相似度會受訓練任務和距離定義影響。 |
| Instruction tuning（指令微調） | 給模型任務指令與對應範例，讓同一個模型理解不同任務的要求。 | LLM-Embedder 用任務指令提示這次要找知識、記憶或其他類型的內容。 |
| LLM feedback、distillation（語言模型回饋、知識蒸餾） | 先讓較強的模型對候選提供回饋，再讓檢索器學它的相對偏好。 | 教學例可比較兩份文件各自讓正確答案變得多可能。 |
| Hard negative（難負樣本） | 看起來和問題很像，但對該任務不適合的候選。 | 問某型號保固，另一型號的保固頁比無關新聞更難排除。 |
| Homogeneous in-batch negatives（同質批內負樣本） | 從同任務的其他訓練例取候選，作為目前查詢的負樣本。 | 先核對候選是否其實也能回答目前查詢，以免把真相關項誤當負樣本。 |

[LLM-Embedder 原論文 v2](https://arxiv.org/abs/2310.07554v2) · [E3 講義](units/e3-retrieval-probability.md)

## P08 RAG Survey：檢索、答案和來源要怎麼評估？

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| BM25（詞彙檢索評分） | 根據查詢詞、文件詞頻、詞在文件庫的稀有程度與文件長度打分。 | Lewis 用固定 BM25 當作稠密檢索的比較條件。 |
| Top-k、Recall@k 與 Hit@k | Top-k 是保留前 k 筆。Recall@k 衡量找回多少已標註相關文件；Hit@k 看這一題有沒有至少命中一份。 | 共有 3 份相關文件、前 2 名命中 1 份，這一題 Recall@2 為 1／3，Hit@2 為 1。 |
| MRR（Mean Reciprocal Rank，平均倒數名次） | 對每題的第一個相關命中取排名倒數，再跨題平均；無命中時記 0。 | 兩題首次命中為第 1 和第 4 名，MRR 是 0.625。 |
| EM（Exact Match，完全匹配） | 依事先約定的答案正規化與比對規則，判斷輸出是否符合參考答案。 | 比較論文時，要核對標點、大小寫和可接受答案的處理。 |
| Reranking（重排） | 先取回一批候選，再用另一套分數重新排序，挑出實際送進生成器的片段。 | 測重排的影響時固定候選池與最後片段數。 |
| Ablation（消融） | 控制其他條件，拿掉或替換一個組件，重做評估來看它的影響。 | 比較有無重排時，固定題目、候選文件和生成器。 |
| Faithfulness、Context relevance、Answer relevance | 依序檢查答案主張是否獲來源支持、取回片段是否關聯問題、以及回答是否切題。 | 來源支持與來源本身的真實性各自需要核對。 |
| Oracle context（人工提供正確證據） | 用已知支持答案的片段，測試生成器在資料充足時的作答。 | 和一般檢索結果配對比較，找出差異比較可能出在哪個階段。 |

[RAG Survey v5](https://arxiv.org/html/2312.10997v5) · [E4 講義](units/e4-rag-ablation.md)

## 應用框架補充

| 名詞 | 怎麼理解 | 讀論文時怎麼用 |
|---|---|---|
| LCEL（LangChain Expression Language） | 用 Runnable 等可組合的單元表達 LangChain 的處理步驟。 | 可依序串接 prompt、模型和輸出解析器。 |
| DAG（Directed Acyclic Graph，有向無環圖） | 節點之間的連線有方向，而且沿著方向走不會回到起點。 | 檢索再回答可用 DAG 表達；在設計回到前一步重試的流程時，需標明循環。 |

[LangChain 可組合流程](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence) · [LangGraph 工作流](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
