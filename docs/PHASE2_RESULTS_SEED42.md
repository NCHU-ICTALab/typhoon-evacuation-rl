# Phase 2 Decomposed PPO：Seed 42 結果

## 設定

- 單一 preference-conditioned actor；
- Preference encoder + FiLM vessel gating；
- count、GT、risk 三頭 critic；
- vector rollout、逐目標 GAE、逐目標 advantage normalization；
- 每個 objective 先完成 PPO clipping，再依 transition preference late scalarize；
- 四個 profile 各 100,000 transitions，總計 400,000；
- 五個 held-out 日期、每 profile 60 個完全配對案例；
- action mask 維持封港、入口與拖船硬限制。

技術理由與公式見 `PHASE2_DECOMPOSED_PPO.md`。這仍是 seed 42 單次研究結果。

## 驗收

| 項目 | Phase 1 | Phase 2 | 判定 |
|---|---:|---:|---|
| Safety violations / rejected | 0 / 0 | 0 / 0 | 通過 |
| 有偏好 action 差異 | 55/60 | 55/60 | 通過，維持 |
| 四種 action trace 全不同 | 35/60 | 33/60 | 略降 |
| Cross-utility 對角最佳 | 1/4 | 1/4（count） | 未改善 |
| GT utility | 0.3480 | 0.3633 | 通過，+0.0153 |
| Risk specialist regret | 2.9% | 3.3% | 未通過 |

## Utility 與 specialist regret

| Preference | Phase 1 utility | Phase 2 utility | Specialist | Phase 2 regret |
|---|---:|---:|---:|---:|
| count | 0.4092 | 0.3929 | 0.4031 | 2.5% |
| balanced | 0.3863 | 0.3821 | 0.3895 | 1.9% |
| GT | 0.3480 | 0.3633 | 0.3683 | 1.4% |
| risk | 0.3929 | 0.3913 | 0.4048 | 3.3% |

分解 critic 明顯修復 GT credit assignment：GT regret 從 5.5% 降至 1.4%。代價是 count、
balanced、risk 小幅退步；同一模型內，balanced、GT、risk utility 仍由 count 輸出取得較高
分數。因此本階段是部分成功，不能宣稱已學成完整 Pareto-conditioned policy。

## Gradient cosine

訓練期間每個 rollout 的第一個 minibatch 量測 objective actor gradient：

| Pair | Mean cosine |
|---|---:|
| count–GT | 0.1860 |
| count–risk | 0.7246 |
| GT–risk | 0.2193 |

平均皆為正值，尤其 count–risk 高度同向；目前沒有充分證據立即套用 PCGrad。最後一個
rollout 的 count–GT 與 count–risk 為負，但單一 rollout 不足以代表整體。後續應保存負
cosine 發生率、分位數與 gradient norm，再決定 gradient surgery。

## 結論與下一步

1. 保留 decomposed critic：它確實改善 GT，不回到 early scalarization。
2. 暫不只增加 steps，也不立刻加入 diversity regularizer。
3. 下一個最小實驗應記錄完整 gradient conflict 分布，並做 actor 結構 ablation：
   shared actor 對照 shared trunk + 四個 preference heads／mixture-of-experts。
4. 若負 cosine 比例高，再加入 PCGrad；若不高，問題較可能是共享 actor 容量或四個線性
   preference 的 policy routing，而不是單純梯度互斥。
5. 在 cross-utility 對角最佳改善前，不切換前端正式模型，也不擴到多 seed。

完整輸出位於被 Git 忽略的：

- `typhoon/models/decomposed_ppo/seed-42/final.pt`
- `typhoon/models/decomposed_ppo/seed-42/training_manifest.json`
- `typhoon/models/decomposed_ppo/seed-42/evaluation.json`
- `typhoon/models/decomposed_ppo/seed-42/comparison.json`
