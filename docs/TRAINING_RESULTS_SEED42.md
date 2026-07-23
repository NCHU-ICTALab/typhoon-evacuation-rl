# Seed 42 正式訓練結果

## 設定

- 訓練日期：24 個；held-out 日期：5 個（2026-07-04 至 2026-07-08）
- 每個模型：100,000 steps、seed 42、MaskablePPO
- 模型：count、balanced、GT、risk 四個 fixed specialist，加一個 Dirichlet-conditioned baseline
- 評估：每 profile 60 個完全配對案例；日期、closure、tugs、compression、jitter、seed 相同
- risk：與 GT 解耦的 deterministic synthetic proxy；GT-risk correlation 0.0396

這是單一 seed 的第一批診斷，不是統計顯著性結論。

## Specialist

| Profile | Own utility | Evacuated count | Evacuated GT | Evacuated risk | Safety/rejected |
|---|---:|---:|---:|---:|---:|
| count | 0.4031 | 12.42 | 302,337 | 25.75 | 0 / 0 |
| balanced | 0.3895 | 12.68 | 286,597 | 25.72 | 0 / 0 |
| GT | 0.3683 | 12.58 | 306,391 | 25.45 | 0 / 0 |
| risk | 0.4048 | 12.32 | 296,604 | 26.37 | 0 / 0 |

60 個 paired cases 中，55 個產生至少兩種不同排程，46 個產生四種不同排程。GT 與 risk
specialist 在對應 utility 欄位為最佳；count 與 balanced 尚未對角最佳，但和最佳值差距小。
這表示環境已能表達目標取捨，但單一 seed/100k 尚不足以證明每個 specialist 收斂。

## Conditioned baseline

| Preference | Conditioned utility | Specialist utility | Relative regret |
|---|---:|---:|---:|
| count | 0.3998 | 0.4031 | 0.8% |
| balanced | 0.3827 | 0.3895 | 1.7% |
| GT | 0.3573 | 0.3683 | 3.0% |
| risk | 0.3878 | 0.4048 | 4.2% |

conditioned 模型只有 20/60 個 paired cases 因偏好而改變 action trace，40/60 完全相同；
cross-utility matrix 的四個對角線均不是各列最佳。它的安全違規與 rejected action 仍為 0，
但偏好敏感度明顯低於 specialist。

## 與規則基線

混合四偏好的平均 weighted utility：conditioned RL 0.3819、FCFS 0.3814、Risk-aware
0.3776、value-density 0.4350。此平均只能看整體表現，不能取代逐偏好的配對結論。

## 下一步決策

1. 不直接增加目前 conditioned PPO steps；先做均衡四偏好 rollout。
2. 加入獨立 preference encoder／feature gating，避免三個權重被 344 維 observation 淹沒。
3. 以相同 seed 42 重訓，若 action separation 與 regret 改善，再擴到 seeds 43、44。
4. 若 Phase 1 仍塌縮，再實作多頭 critic、逐目標 GAE 的 D3PO-lite。

完整 JSON 位於被 Git 忽略的：

- `typhoon/models/specialists/comparison-seed-42.json`
- `typhoon/models/evaluation.json`
