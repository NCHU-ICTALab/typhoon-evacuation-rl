# Phase 9A：Frozen Experts + Continuous Preference Soft Router

## 目的與定位

Phase 8 已有四個通過 GT/risk gate 的 fixed-profile experts。Phase 9A 先把它們封裝進同一個
checkpoint，凍結 expert 參數，只訓練連續偏好 router。這是從「四個離散模型」走向
「同一模型、連續偏好」的低風險過渡版，不是最終 end-to-end MORL 模型。

前端預設仍使用 Phase 8 hard routing；Phase 9A 必須由「實驗性連續偏好」開關啟用，避免
尚未完成的 count／balanced 特化被誤認為正式版本。

## 模型結構

- 單一 `final.pt` 內含 count、balanced、GT、risk 四個 frozen Phase 8 experts。
- preference router 接收三維權重 `(count, GT, risk)`，輸出四個 expert 的 softmax 權重。
- state router 已保留介面，但最後一層初始化為 0，Phase 9A 尚未用 PPO 訓練。
- 每個 expert 仍以自己訓練時的 centroid observation 推論；連續偏好只供 router 使用，避免
  fixed-profile expert 被送到未見過的偏好分布。
- 最終 action distribution 是四個 masked expert action probabilities 的加權和。無效動作在
  混合後再次清為精確 0，安全限制不交給 router 學習。
- critic 從 Phase 8 balanced expert 初始化，供下一階段 router-PPO／joint fine-tuning 使用。

## Router 訓練

執行：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.train_soft_moe \
  --db /path/to/ua1008l.sqlite --seed 42
~~~

Router 使用一半 Dirichlet(0.7) 連續偏好、一半四個 centroid，依偏好與 centroid 的平方距離
建立 RBF soft target。Seed 42 使用 2,000 steps、batch 512、learning rate 0.003、temperature
0.02；cross-entropy 由 1.3743 降至 0.0624。這是 router distillation，不是新的 RL rollout。

Checkpoint 位於 `typhoon/models/soft_moe_router_distilled/seed-42/final.pt`，大小約 7.2 MB，
依專案規則不進 Git；可追蹤摘要位於 `typhoon/model_cards/phase9a_seed42.json`。

## Seed 42 held-out 結果

四個 centroid 的目標 expert 權重皆超過 99.98%。同一 checkpoint 在 60 個 paired base cases
上得到：

| 偏好 | utility | 對角最佳 |
|---|---:|:---:|
| count | 0.4018 | 否 |
| balanced | 0.3813 | 否 |
| GT | 0.3669 | 是 |
| risk | 0.4005 | 是 |

- GT/risk gate：2/2 通過。
- 安全違規／rejected actions：0／0。
- 56/60 cases 會因偏好改變 action trace；50/60 cases 有四條不同 action trace。
- count／balanced 未達對角最佳，與 Phase 8 expert 本身的限制一致，router 並未修復它。

## 真實 API smoke record

保留日 `2026-07-08`，封港 4 小時、拖船 8、需求壓力 3，自訂偏好
`(count=0.2, GT=0.3, risk=0.5)`：

- API health／schedule：200／200。
- engine：`python-phase8-experts+phase9a-soft-moe`。
- 初始 router（count/balanced/GT/risk）：`0.000000 / 0.719629 / 0.000012 / 0.280359`。
- 排程 KPI：9 艘、200,376 GT、16 風險點；安全違規 0、rejected actions 0。

Router 權重不必等於輸入的三個目標權重；它代表四個 expert 的混合比例。此例較接近 balanced
centroid，因此 balanced expert 權重較高。

## 下一階段驗收

Phase 9B 才進行 router RL：先凍結 experts，以向量 critic/GAE 依偏好合成 scalar advantage，
對 router 做單次 PPO clipping；通過 gate 後才以小 learning rate 解凍 expert 上層。每一步都
必須與 Phase 8 對照，至少維持安全 0、GT/risk 2/2，並檢查 centroid 間 preference grid 的
平滑度、Pareto coverage 與 regret。正式展示前還要補多 seed、GT-aware baseline 與最佳化上限。
