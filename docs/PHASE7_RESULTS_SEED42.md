# Phase 7 Preference-first surrogate：Seed 42 結果

## 驗收摘要

| 項目 | Phase 6 per-objective clip | Phase 7 preference-first |
|---|---:|---:|
| 每 profile transitions | 100,000 | 100,000 |
| Safety violations / rejected | 0 / 0 | 0 / 0 |
| 四種 action trace 全不同 | 51/60 | **51/60** |
| 有多種 schedule | 58/60 | **60/60** |
| Cross-utility 對角最佳 | 1/4（balanced） | **2/4（count、GT）** |
| GT、risk 對角最佳 | 0/2 | **1/2（GT）** |

先依偏好合成 scalar advantage，再做一次 normalization 與 PPO clipping，成功恢復 GT
對角最佳，並讓 count 也成為對角最佳；但 risk 仍未恢復，因此 soft MoE gate 尚未完全
通過。

## Utility 比較

| Fixed profile | Phase 6 | Phase 7 | Delta | 舊 SB3 specialist | Phase 7 vs SB3 |
|---|---:|---:|---:|---:|---:|
| count | 0.3943 | **0.4002** | +0.0059 | 0.4031 | -0.0029 |
| balanced | **0.3897** | 0.3745 | -0.0152 | 0.3895 | -0.0150 |
| GT | 0.3499 | **0.3594** | +0.0095 | 0.3683 | -0.0089 |
| risk | **0.3931** | 0.3859 | -0.0072 | 0.4048 | -0.0189 |

這項修正明顯幫助 count 與 GT，但不是所有偏好都單調改善；balanced 與 risk 退步，表示
逐目標 clipping 是 GT 錯位的重要原因，卻不是 risk 失敗的唯一原因。

## Phase 7 cross-utility

列是評分偏好，欄是產生排程的固定模型：

| 評分偏好 \ Model | count | balanced | GT | risk |
|---|---:|---:|---:|---:|
| count | **0.400196** | 0.392803 | 0.394329 | 0.388521 |
| balanced | **0.386855** | 0.374500 | 0.380607 | 0.376529 |
| GT | 0.358866 | 0.345114 | **0.359428** | 0.355212 |
| risk | **0.401502** | 0.385584 | 0.388064 | 0.385854 |

GT model 只以 0.000562 的差距勝過 count model，但已在相同 60 個 paired cases 上達成
對角最佳。risk model 則落後 count model 0.015648，不能視為估計誤差範圍內的小差距。

## 三目標成分

由 cross-utility 線性反解 normalized completion：

| Fixed model | Count | GT | Risk |
|---|---:|---:|---:|
| count | **0.4111** | 0.3360 | **0.4135** |
| balanced | 0.4078 | 0.3211 | 0.3947 |
| GT | 0.4056 | **0.3421** | 0.3942 |
| risk | 0.3983 | 0.3378 | 0.3935 |

GT model 的 GT 成分已正確最高；risk model 的 risk 成分反而最低，risk 錯位仍是實質的
policy learning 問題，不是 cross-utility 標籤或 router 映射錯誤。

## 梯度衝突

| Profile | count–GT 負 cosine | count–risk | GT–risk |
|---|---:|---:|---:|
| count | 35.7% | 7.7% | 38.8% |
| balanced | 41.3% | 5.6% | 40.8% |
| GT | 38.8% | 6.1% | 33.7% |
| risk | 41.8% | 5.6% | 38.8% |

preference-first 沒有消除目標衝突，也不應消除；它只避免三個 objective 各自選擇不同的
PPO clipping 分支。GT 的改善支持「clipping 順序有影響」，但負 cosine 仍代表真實排程
取捨。

## 問題定位與下一步

固定 preference 下，若 scalar value 等於三頭 value 的偏好加權，vector GAE 的偏好加權在
代數上等價於 scalar GAE。因此 Phase 7 已排除 actor scalarization 公式作為 risk 唯一原因。
與舊 SB3 specialists 尚有三項主要差異：

1. custom critic 對三個 value heads 做等權 MSE；SB3 只訓練該偏好的 scalar value；
2. custom advantage 以整個 rollout normalization；SB3 在 minibatch 內 normalization；
3. custom 使用 preference-gated 256/128 extractor；SB3 使用一般 128/128 MLP。

下一個最小受控實驗應先改 critic loss：保留 vector critic/returns，但以 preference-scalarized
value MSE 作主要 loss，另留小權重 vector auxiliary loss。固定其他條件後先只重跑 risk 與
GT；若 risk 恢復且 GT 不退步，再重跑四 profile，之後才進入 soft MoE。

完整輸出位於被 Git 忽略的：

- `typhoon/models/decomposed_ppo_preference_first_fixed_profiles/{profile}/seed-42/final.pt`
- `typhoon/models/decomposed_ppo_preference_first_fixed_profiles/{profile}/seed-42/training_manifest.json`
- `typhoon/models/decomposed_ppo_preference_first_fixed_profiles/comparison-seed-42.json`

後續 Phase 8 已完成上述 preference-scalarized critic 實驗，並恢復 GT、risk 2/2 gate；結果
見 `PHASE8_RESULTS_SEED42.md`。
