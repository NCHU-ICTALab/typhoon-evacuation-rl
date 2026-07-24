# Phase 4：Shared trunk + hard preference heads

## 目的

Phase 3 顯示 PCGrad 會移除 MORL 中有意義的目標取捨，且沒有改善 cross-utility。Phase 4
改測另一個假說：單一 actor output layer 是否無法把 preference 映射到正確的 policy region。

本階段維持一個 checkpoint、共享的 preference-gated feature extractor、actor trunk 與三頭
critic，只將最後的 action logits 分成 count、balanced、GT、risk 四個 head。

## Routing

四個訓練 profile 分別路由到固定 head：

| Profile | Preference weights | Head |
|---|---|---:|
| count | `(0.70, 0.15, 0.15)` | 0 |
| balanced | `(1/3, 1/3, 1/3)` | 1 |
| GT | `(0.15, 0.70, 0.15)` | 2 |
| risk | `(0.15, 0.15, 0.70)` | 3 |

實作以 preference 到四個 profile centroid 的最短距離選 head。此行為適合四種離散 PoC
偏好，但尚不是可對任意連續權重平滑插值的 Pareto policy。

## 受控條件

- 四個 heads 從完全相同的權重起跑；
- 第一個 head、shared trunk 與 critic 的初始化順序保持與 Phase 2 一致；
- PCGrad 關閉；
- vector reward、per-objective GAE/PPO clipping、late scalarization 與 equal critic 不變；
- 四 profile 各 100,000 transitions，paired scenario stream 與 seed 42 不變；
- 30 艘、31 個動作及 hard action mask 不變；
- 輸出到獨立的 `decomposed_ppo_hard_heads/`。

## 命令

~~~bash
.venv/bin/python -m typhoon.train_decomposed_ppo \
  --db data/ua1008l.sqlite \
  --steps-per-profile 100000 \
  --seed 42 \
  --actor-routing hard_heads \
  --gradient-surgery none
~~~

## 判讀

這是 routing 診斷，不是最終 MORL 架構。成功至少應恢復 specialist 已證明可達的 GT、risk
對角最佳，並降低相對 specialist regret。count、balanced 在四個 specialist 實驗中本來就
不是對角最佳，因此 4/4 只能當探索目標，不能當作已證明可達的必要條件。
