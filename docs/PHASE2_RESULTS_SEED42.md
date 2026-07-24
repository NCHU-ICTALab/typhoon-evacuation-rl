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

訓練平均值只能說明整體方向，無法顯示 rollout 間的正負切換。因此另以 seed 42 最終
checkpoint 做 post-hoc 診斷：凍結模型、不更新參數，重新抽取平衡的四偏好 rollout；
共 50 個 rollout、每次 64 steps × 4 profiles，合計 12,800 transitions。每個 scope 有
50 個 cosine 樣本，梯度由 normalized objective advantage 的 actor loss 計算。

### Aggregate 分布

| Pair | Mean | Std | Min | P05 | P25 | Median | P75 | P95 | Max | 負 cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| count–GT | 0.177 | 0.539 | -0.878 | -0.687 | -0.323 | 0.314 | 0.635 | 0.868 | 0.965 | 38% |
| count–risk | 0.717 | 0.258 | -0.009 | 0.137 | 0.583 | 0.812 | 0.903 | 0.962 | 0.994 | 2% |
| GT–risk | 0.208 | 0.477 | -0.800 | -0.472 | -0.216 | 0.295 | 0.600 | 0.857 | 0.924 | 40% |

平均 cosine 為正確實掩蓋了局部衝突：count–GT 與 GT–risk 的負值比例分別是 38% 與
40%，且 P25 已落在負值；相反地，count–risk 幾乎總是同向，負值只有 2%。

### 各偏好下的負 cosine 比例

| Rollout profile | count–GT | count–risk | GT–risk |
|---|---:|---:|---:|
| count | 42% | 10% | 30% |
| balanced | 32% | 10% | 28% |
| GT | 44% | 0% | 38% |
| risk | 34% | 8% | 24% |

衝突在 GT profile 最明顯：count–GT 為 44%、GT–risk 為 38%。這與 Phase 2 中 GT
需要 decomposed credit assignment 才明顯改善的結果一致。balanced profile 也有 32%
與 28% 的 GT 相關衝突，可能阻礙單一輸出同時成為 balanced utility 的對角最佳。

### Gradient norm

| Objective | Mean norm | Std |
|---|---:|---:|
| count | 1.034 | 0.427 |
| GT | 1.010 | 0.346 |
| risk | 0.984 | 0.400 |

三個 objective 的平均 gradient norm 接近，因此目前觀察到的負 cosine 不能只解釋為
某一目標梯度量級特別大。需注意這是 final policy 的 post-hoc、seed 42 單次診斷；
足以支持下一個受控消融，但還不是跨 seed 的統計結論。

## 結論與下一步

1. 保留 decomposed critic：它確實改善 GT，不回到 early scalarization。
2. 暫不只增加 steps，也不立刻加入 diversity regularizer。
3. 下一個最小實驗改為同一 seed、同一資料順序與超參數下，比較「目前 decomposed PPO」
   與「shared actor/extractor 加 PCGrad」；PCGrad 僅在 objective dot product 為負時投影。
4. 主要假說是處理 count–GT、GT–risk 的局部衝突；count–risk 高度同向，不應被當作
   需要拆解的主要問題。應同時追蹤 utility、specialist regret、cross-utility 對角最佳與
   負 cosine 比例，避免只改善梯度指標。
5. 若 PCGrad 無法改善 balanced/risk routing，再做 shared trunk + preference heads 或
   mixture-of-experts 的 actor 結構 ablation。
6. 在 cross-utility 對角最佳改善前，不切換前端正式模型，也不直接擴到多 seed；先確認
   seed 42 的受控消融是否有方向性效果。

完整輸出位於被 Git 忽略的：

- `typhoon/models/decomposed_ppo/seed-42/final.pt`
- `typhoon/models/decomposed_ppo/seed-42/training_manifest.json`
- `typhoon/models/decomposed_ppo/seed-42/evaluation.json`
- `typhoon/models/decomposed_ppo/seed-42/comparison.json`
- `typhoon/models/decomposed_ppo/seed-42/gradient_cosine_diagnostic.json`
