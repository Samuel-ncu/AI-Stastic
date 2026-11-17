# AI-Stastic

## AI 深度學習的統計基礎與實驗判讀

配合 [NVIDIA DLI 深度學習基礎中文教材](https://github.com/NVDLI/fundamentals-of-deep-learning-zh) 的繁體中文補充教學專案，適合商管及跨領域學生。

學習目標：看懂資料、理解模型如何學習、判斷結果是否可信，並以統計證據說明 AI 模型的限制。

## 從這裡開始

- [課程大綱與時數](docs/syllabus.md)
- [DLI 章節對照](docs/dli-mapping.md)
- [十個統計單元](docs/units/README.md)
- [三個核心實驗](exercises/README.md)
- [教師教學規劃](instructor/README.md)
- [教材開發路線](docs/roadmap.md)
- [資料來源與使用規範](data/README.md)
- [參考來源與署名](docs/references.md)
- [貢獻方式](CONTRIBUTING.md)

## 教學方式

問題情境 → 圖像直覺 → 最少必要公式 → Python 實驗 → 結果解讀 → DLI 練習。

每次作業須同時繳交方法、結果、解讀及限制。AI 可以協助產生程式，但學生須驗證輸出並解釋統計意義。

## 立即開始

已提供 [10 份統計教學 Notebook 與 Colab 入口](notebooks/README.md)，每份包含概念、Python 範例、結果解讀與練習。只需 CPU，使用合成或內建資料。

[查看執行驗證](https://github.com/Samuel-ncu/AI-Stastic/actions/workflows/notebooks.yml)。CI 會在全新 Python 環境執行全部 Notebook；執行完成的版本可從 Actions artifacts 下載。

目前是統計入門教材；完整 DLI 模型對照實驗與三個核心專題的整合實作仍待開發。

## 專案目錄

| 目錄 | 用途 |
|---|---|
| docs/ | 課綱、章節對照及單元教材綱要 |
| exercises/ | 學生實驗與作業 |
| instructor/ | 教學提示與評分規準；目前公開，不放需保密解答 |
| notebooks/ | 10 份統計教學 Notebook 與 Colab 入口 |
| src/ | 待開發共用程式的規格 |
| data/ | 資料來源及下載規範 |

本專案為獨立補充教材，非 NVIDIA 官方課程或認證。原課程以連結引用；不包含 NVIDIA 投影片、資料集或認證測驗答案。
