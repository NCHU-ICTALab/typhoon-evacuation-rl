# Phase 5 Full actor experts：Seed 42 結果

## 設定

- 固定 30 艘、31 個離散動作；
- shared preference-gated extractor；
- count、balanced、GT、risk 四個相同初始化的完整 256/128 actor experts；
- shared three-head critic，PCGrad 關閉；
- 四 profile 各 100,000 transitions，總計 400,000；
- 五個 held-out 日期、每 profile 60 個完全配對案例。

技術契約見 `PHASE5_FULL_EXPERTS.md`。本結果仍是 seed 42 單次結構消融。

## 驗收摘要

| 項目 | Phase 2 | Hard heads | Full experts |
|---|---:|---:|---:|
| Safety violations / rejected | 0 / 0 | 0 / 0 | 0 / 0 |
| 有偏好 action 差異 | 55/60 | 59/60 | 55/60 |
| 四種 action trace 全不同 | 33/60 | 52/60 | 50/60 |
| Cross-utility 對角最佳 | 1/4 | 1/4 | **0/4** |
| GT、risk 對角最佳 | 0/2 | 0/2 | **0/2** |

完整 actor experts 仍產生高度不同的排程，但沒有恢復 specialist 已證明可達的 GT、risk
對角。共享 actor trunk 容量不足不是主要解釋。

## Utility 與 specialist regret

| Preference | Phase 2 | Hard heads | Full experts | Specialist | Full-expert regret |
|---|---:|---:|---:|---:|---:|
| count | 0.3929 | 0.4068 | **0.4057** | 0.4031 | -0.7% |
| balanced | 0.3821 | 0.3814 | **0.3840** | 0.3895 | 1.4% |
| GT | **0.3633** | 0.3629 | 0.3581 | 0.3683 | 2.8% |
| risk | 0.3913 | 0.3875 | **0.3940** | 0.4048 | 2.7% |

risk utility 與 regret 比 Phase 2 改善，但 risk expert 並不是 risk 評分下的最佳輸出；因此
不能只看自己的 utility，就宣稱 preference routing 成功。

## Cross-utility matrix

列是評分偏好，欄是產生排程的 expert：

| 評分偏好 \ Expert | count | balanced | GT | risk |
|---|---:|---:|---:|---:|
| count | 0.405699 | 0.401066 | **0.408489** | 0.395803 |
| balanced | **0.395010** | 0.384036 | 0.388987 | 0.387279 |
| GT | 0.369265 | 0.355757 | 0.358070 | **0.372022** |
| risk | **0.410066** | 0.395286 | 0.400402 | 0.394010 |

四列最佳都不在對角線：GT expert 在 count 評分最高、risk expert 在 GT 評分最高、count
expert 在 risk 評分最高，形成循環錯位。

## 三目標成分

由 cross-utility 線性反解 normalized completion：

| Producing expert | Count | GT | Risk |
|---|---:|---:|---:|
| count | 0.4144 | 0.3482 | **0.4224** |
| balanced | 0.4150 | 0.3326 | 0.4045 |
| GT | **0.4244** | 0.3328 | 0.4097 |
| risk | 0.4028 | **0.3595** | 0.3995 |

這與 preference 名稱呈現同樣錯位：GT expert 的 count 成分最高、risk expert 的 GT 成分最高、
count expert 的 risk 成分最高。

## 映射稽核

已逐一核對，沒有發現 label wiring bug：

- profile 建立順序為 count、balanced、GT、risk；
- observation preference、objective reward 與 utility 欄位順序皆為 count、GT、risk；
- routing centroid 與四個 profile 權重完全一致；
- checkpoint manifest 記錄 `actor_routing=full_experts`；
- safety violations 與 rejected actions 皆為 0。

因此循環錯位是模型／optimizer 的學習結果，不是 expert 名稱接反。

## 結論與下一步

1. 不切換前端模型，也不直接加入 soft MoE；混合四個錯位 experts 不會自然得到正確 Pareto
   routing。
2. full experts 排除了「只因共享 actor trunk 容量不足」的主要假說。
3. 下一個最小診斷應訓練四個 **custom decomposed PPO fixed-profile models**：使用本專案同一
   vector critic、GAE、late scalarization 與 optimizer，但每次只訓練一個 profile。
4. 若 custom fixed-profile 能恢復 GT、risk 對角，問題在共享 extractor／critic 或 multi-profile
   optimizer；若仍不能，問題在目前 decomposed PPO objective/update，而不是 routing 容量。
5. 舊 SB3 specialists 使用 scalar critic、不同網路與隨機情境流，只能證明環境有 trade-off，
   不能直接定位本 custom optimizer 的故障層。
6. 在完成 fixed-profile 同演算法對照前，不再增加 actor 分支或訓練步數。

完整輸出位於被 Git 忽略的：

- `typhoon/models/decomposed_ppo_full_experts/seed-42/final.pt`
- `typhoon/models/decomposed_ppo_full_experts/seed-42/training_manifest.json`
- `typhoon/models/decomposed_ppo_full_experts/seed-42/evaluation.json`
- `typhoon/models/decomposed_ppo_full_experts/seed-42/comparison.json`
