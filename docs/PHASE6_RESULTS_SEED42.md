# Phase 6 Custom fixed-profile：Seed 42 結果

## 驗收摘要

| 項目 | 舊 SB3 specialists | Custom fixed-profile |
|---|---:|---:|
| 每 profile transitions | 100,000 | 100,000 |
| Safety violations / rejected | 0 / 0 | 0 / 0 |
| 四種 action trace 全不同 | 46/60 | **51/60** |
| 有多種 schedule | 55/60 | **58/60** |
| Cross-utility 對角最佳 | 2/4（GT、risk） | **1/4（balanced）** |
| GT、risk 對角最佳 | 2/2 | **0/2** |

完全切斷多偏好參數共享後，custom decomposed PPO 仍未恢復 GT、risk。Soft MoE gate
未通過。

## Utility 比較

| Fixed profile | 舊 SB3 specialist | Custom fixed | Delta |
|---|---:|---:|---:|
| count | 0.4031 | 0.3943 | -0.0088 |
| balanced | 0.3895 | **0.3897** | +0.0002 |
| GT | 0.3683 | 0.3499 | -0.0184 |
| risk | 0.4048 | 0.3931 | -0.0117 |

Custom 版本只在 balanced 接近舊 specialist；GT 退步最大。

## Custom fixed-profile cross-utility

列是評分偏好，欄是產生排程的固定模型：

| 評分偏好 \ Model | count | balanced | GT | risk |
|---|---:|---:|---:|---:|
| count | 0.394311 | **0.403022** | 0.401675 | 0.391390 |
| balanced | 0.374456 | **0.389741** | 0.379956 | 0.383583 |
| GT | 0.336044 | 0.361911 | 0.349949 | **0.366281** |
| risk | 0.393014 | **0.404290** | 0.388244 | 0.393079 |

GT 評分由 risk model 最高，risk 評分由 balanced model 最高；不是對角 routing 問題，因為
這四顆模型彼此完全獨立。

## 三目標成分

由 cross-utility 線性反解 normalized completion：

| Fixed model | Count | GT | Risk |
|---|---:|---:|---:|
| count | 0.4106 | 0.3046 | 0.4082 |
| balanced | 0.4139 | 0.3391 | **0.4162** |
| GT | **0.4194** | 0.3254 | 0.3950 |
| risk | 0.3978 | **0.3521** | 0.4008 |

GT model 的 count 成分最高，risk model 的 GT 成分最高，與 Phase 5 的循環錯位方向一致。

## 固定偏好下的 gradient conflict

| Profile | count–GT 負 cosine | count–risk | GT–risk |
|---|---:|---:|---:|
| count | 36.2% | 5.1% | 40.3% |
| balanced | 34.2% | 3.6% | 33.7% |
| GT | 36.7% | 3.6% | 34.7% |
| risk | 36.7% | 6.1% | 35.2% |

即使固定單一 preference，GT 相關 objective gradients 仍約三分之一時間為負。這表示衝突
主要是同一排程問題內的目標取捨，不是四 profile 共用模型才產生。

## 問題定位

舊 SB3 specialists 使用 scalar reward → scalar GAE → 單一 PPO clipping；custom 版本使用
vector GAE → per-objective normalization → per-objective PPO clipping → preference weighting。
兩者的環境與 fixed preference 相同，但 custom 極端偏好明顯較差。

最可疑的差異是 PPO clipping 的非線性順序：

~~~text
目前：每個 objective 各自 min(unclipped, clipped)，再加權
對照：先將 vector advantage 依 preference scalarize，再做一次 PPO clipping
~~~

`min` 與加權不可交換；per-objective clipping 可能扭曲極端偏好的 aggregate policy gradient。

## Soft MoE 前停止結論

1. 不加入 soft MoE：目前沒有通過 GT、risk gate 的 custom experts 可供混合。
2. 已排除 action head、共享 actor trunk、多偏好共享 extractor／critic 為主要單一原因。
3. 下一個最小演算法修正應保留 vector critic/GAE，但將 actor 改成：先 scalarize vector
   advantage，再 normalization 與單一 PPO clipping。
4. 應先在四個 fixed profiles 重跑這個 surrogate ablation；至少恢復舊 specialists 的 GT、risk
   2/4，才重新開啟 soft MoE。
5. 本階段停在 soft MoE 前的明確 gate，避免用更複雜架構掩蓋 optimizer/objective 問題。

完整輸出位於被 Git 忽略的：

- `typhoon/models/decomposed_ppo_fixed_profiles/{profile}/seed-42/final.pt`
- `typhoon/models/decomposed_ppo_fixed_profiles/{profile}/seed-42/training_manifest.json`
- `typhoon/models/decomposed_ppo_fixed_profiles/comparison-seed-42.json`
