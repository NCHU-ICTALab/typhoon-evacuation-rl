# Phase 8 Preference-scalarized critic：Seed 42 結果

## Gate 結論

| 項目 | Phase 7 | Phase 8 |
|---|---:|---:|
| 每 profile transitions | 100,000 | 100,000 |
| Safety violations / rejected | 0 / 0 | 0 / 0 |
| 四種 action trace 全不同 | 51/60 | 50/60 |
| 有多種 schedule | 60/60 | 55/60 |
| Cross-utility 對角最佳 | count、GT | **GT、risk** |
| GT、risk gate | 1/2 | **2/2，通過** |

Phase 8 恢復舊 SB3 specialists 已證明可學的兩個極端偏好。Experts 不再出現 GT model
偏 count、risk model 偏 GT 的循環錯位，因此 soft MoE 的 expert gate 正式通過。

## Utility 比較

| Fixed profile | Phase 7 | Phase 8 | Delta | 舊 SB3 | Phase 8 vs SB3 |
|---|---:|---:|---:|---:|---:|
| count | 0.4002 | **0.4018** | +0.0016 | 0.4031 | -0.0013 |
| balanced | 0.3745 | **0.3813** | +0.0068 | 0.3895 | -0.0082 |
| GT | 0.3594 | **0.3669** | +0.0075 | 0.3683 | -0.0014 |
| risk | 0.3859 | **0.4005** | +0.0146 | 0.4048 | -0.0043 |

四個 profile 都優於 Phase 7，GT 與 risk 已接近舊 SB3 specialist。risk 的改善最大，支持
「等權 vector critic 沒有把 value approximation 對準極端偏好」是主要原因之一。

## Phase 8 cross-utility

列是評分偏好，欄是產生排程的固定模型：

| 評分偏好 \ Model | count | balanced | GT | risk |
|---|---:|---:|---:|---:|
| count | 0.401774 | 0.395261 | 0.401451 | **0.403680** |
| balanced | 0.387647 | 0.381321 | 0.386249 | **0.388486** |
| GT | 0.363751 | 0.363329 | **0.366927** | 0.361236 |
| risk | 0.397415 | 0.385373 | 0.390369 | **0.400542** |

GT 與 risk 的 regret 都是 0。count 的 regret 為 0.001906，balanced 為 0.007165；這兩者
未對角不阻擋 gate，因為舊 SB3 specialists 的 count、balanced 也未對角最佳。

## 三目標成分

| Fixed model | Count | GT | Risk |
|---|---:|---:|---:|
| count | 0.4133 | 0.3442 | 0.4054 |
| balanced | 0.4067 | 0.3486 | 0.3887 |
| GT | 0.4139 | **0.3511** | 0.3937 |
| risk | **0.4161** | 0.3389 | **0.4104** |

GT model 的 GT 成分最高，risk model 的 risk 成分最高，目標方向已正確。risk model 同時有
最高 count 成分，因此也在 count、balanced 評分下勝出；這是 Pareto 表現，不是 label
wiring 錯誤。

## 梯度衝突仍存在

| Profile | count–GT 負 cosine | count–risk | GT–risk |
|---|---:|---:|---:|
| count | 36.7% | 9.2% | 43.4% |
| balanced | 46.4% | 7.7% | 47.4% |
| GT | 40.8% | 5.6% | 40.3% |
| risk | 38.8% | 7.1% | 36.7% |

Gate 通過不是因為梯度衝突消失，而是 actor clipping 與 critic credit assignment 都對準同一
偏好。這再次說明負 cosine 多半是真實 MORL 取捨，不應直接以 PCGrad 全部消除。

## Soft MoE 下一步

現在可以使用這四顆 fixed-profile experts 作為 soft MoE 的初始化與 teacher baseline。第一版
router 應保持簡單：輸入 preference 與 global state，輸出四 expert weights；先凍結 experts
只訓練 router，再小 learning rate joint fine-tune。驗收必須保留 Phase 8 的 GT/risk 2/2，
並檢查連續偏好插值，而不只四個 centroid。

目前仍只有 seed 42。soft MoE 架構確定後，正式結論至少再跑 3 seeds；在此之前 Phase 8
應視為通過 PoC gate，而不是統計上已定案。

完整輸出位於被 Git 忽略的：

- `typhoon/models/decomposed_ppo_preference_first_scalar_critic_fixed_profiles/{profile}/seed-42/final.pt`
- `typhoon/models/decomposed_ppo_preference_first_scalar_critic_fixed_profiles/{profile}/seed-42/training_manifest.json`
- `typhoon/models/decomposed_ppo_preference_first_scalar_critic_fixed_profiles/comparison-seed-42.json`
