# Phase 5：Shared extractor + full actor experts

## 目的

Phase 4 的四個 output heads 已讓 52/60 cases 產生四種不同 action traces，但仍只有 count
對角最佳。本階段檢查共享的 256/128 actor trunk 是否限制各偏好進入不同 policy region。

模型仍是一個 checkpoint，共享 preference-gated vessel extractor 與三頭 vector critic；
count、balanced、GT、risk 各自擁有完整的 `256 → 128 → 31 actions` actor expert。

## 受控條件

- 四個 experts 從完全相同參數起跑；
- 第一個 expert、extractor 與 critic 的初始化順序維持 Phase 2 基準；
- routing centroid、vector reward、GAE、PPO clipping 與 late scalarization 不變；
- PCGrad 關閉；
- 四 profile 各 100,000 transitions，paired scenario stream 與 seed 42 不變；
- 固定 30 艘、31 個動作與 hard action mask；
- 輸出到獨立的 `decomposed_ppo_full_experts/`。

## 命令

~~~bash
.venv/bin/python -m typhoon.train_decomposed_ppo \
  --db data/ua1008l.sqlite \
  --steps-per-profile 100000 \
  --seed 42 \
  --actor-routing full_experts \
  --gradient-surgery none
~~~

## 成功門檻

四個舊 SB3 specialists 只證明 GT、risk 兩列可對角最佳，因此本階段主要門檻是恢復這
2/4，並讓 GT、risk specialist regret 不高於 Phase 2。count、balanced 的對角最佳仍作為
探索結果，不當成已證明可達的必要條件。
