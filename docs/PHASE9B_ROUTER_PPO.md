# Phase 9B：Frozen-Expert Soft Router PPO

## 目的

Phase 9A 只用偏好距離蒸餾 router，尚未透過環境 reward 學習。Phase 9B 保留同一個 Soft MoE
checkpoint 與四個 frozen Phase 8 experts，使用真實環境 rollout 更新 preference router、
state router 與向量 critic。這是「同一 checkpoint、連續偏好、RL-trained router」的第一版，
但還不是 joint end-to-end fine-tuning。

## PPO 契約

每個 rollout 保留 count、GT、risk 三維 reward、value、return 與 GAE。Actor update 的順序為：

1. 先以 transition 當下的三維偏好合成 scalar advantage；
2. 對 scalar advantage 做一次 normalization；
3. 對 Soft MoE action-mixture probability 做一次 PPO clipping；
4. critic 主 loss 使用偏好合成 value MSE，另加 0.1 倍 vector MSE；
5. 加入小幅 router anchor cross-entropy，避免四個已驗證 centroid 在初期崩掉。

Action mask 在每個 frozen expert 內套用，混合後再次把 invalid actions 清成精確 0。Experts
不在 optimizer 中，測試亦逐參數確認 PPO update 前後完全相同。

## 訓練資料

四個平行 lane 分別對應 count、balanced、GT、risk centroid。每次 episode：

- 50% 使用該 lane 的精確 centroid；
- 50% 使用 Dirichlet(0.7) 連續偏好；
- 四個 lane 使用相同 episode index 的情境、封港時間、拖船與需求壓力 stream。

Seed 42 pilot 使用 25,000 steps/profile，共 100,000 RL transitions；router LR `3e-4`、critic LR
`3e-4`、5 epochs、anchor coefficient `0.01`。執行：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.train_soft_moe_ppo \
  --db /path/to/ua1008l.sqlite --seed 42 --steps-per-profile 25000

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.evaluate_soft_moe_grid \
  --db /path/to/ua1008l.sqlite --step 0.25 --seed 42
~~~

## Seed 42 結果

### 四個 centroid

Phase 9B 與 Phase 9A 的 deterministic KPI／utility 完全相同，GT 與 risk 維持 cross-utility
對角最佳；安全違規與 rejected actions 均為 0。這代表 anchor 成功保住已通過的 expert gate，
但 centroid 本身不能證明 router PPO 有價值。

### 連續偏好 grid

在 15 個 simplex grid 點、每點 60 個 paired held-out cases 上，Phase 9A 與 9B 各執行 900
episodes：

| 指標 | Phase 9B vs 9A |
|---|---:|
| 改善／持平／退步 grid 點 | 4／9／2 |
| 平均 utility delta | +0.001626 |
| 最小 utility delta | -0.000813 |
| 最大 utility delta | +0.015504 |
| 改變 action／schedule 的案例 | 262／900 |
| safety violations／rejected actions | 0／0 |

最大改善出現在 `(count=0.5, GT=0.0, risk=0.5)`，平均 utility 增加 0.015504；另有兩個
中間偏好小幅退步。Pilot gate 定義為平均 delta 不為負、最差 delta 不低於 -0.002、GT/risk
centroid gate 通過且安全為 0，本 checkpoint 通過，因此可供前端「實驗性連續偏好」使用。

## 解讀與下一步

這次證明 PPO 不只改變參數：262 個 held-out cases 的 deterministic schedule 確實改變，連續
grid 平均效用亦轉正。但它仍不能取代 Phase 8 預設模式，原因是只有單 seed、grid 解析度為
0.25、兩個點仍退步，而且 count／balanced 仍非對角最佳。

下一步先重跑 seeds 43、44，再將 grid 細化到 0.1 並加入 bootstrap CI／per-point regret。
只有多 seed 仍維持 gate，才延長到 100k steps/profile 或進入小 learning-rate joint
fine-tuning；不應直接解凍 experts。
