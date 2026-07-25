# Phase 10：偏好單調性（連續偏好下的 GT 排序異常）

## 目的

前端「實驗性連續偏好」模式下，操作人員回報：把偏好從「艘數優先」調到「GT 優先」時，撤離
GT 有時不升反降。本 phase 定義、量化並修復這個 **偏好單調性** 問題：沿 simplex edge 提高某
目標的權重時，該目標的撤離量不應在同一 case 上下降。experts 全程凍結，安全 mask 不變。

## 問題定義與 Phase 9B baseline（seed-42, held-out）

四個 centroid 的 KPI 差異極小，代表 experts 在各目標軸上分化不足：

| 偏好 | 撤離艘數 | 撤離 GT | 撤離風險點 | 自身 utility |
|---|---:|---:|---:|---:|
| count | 12.40 | 298,630 | 25.62 | 0.4018 |
| balanced | 12.20 | 305,372 | 24.50 | 0.3813 |
| gt | 12.42 | 308,200 | 24.92 | 0.3669 |
| risk | 12.48 | 291,572 | 25.92 | 0.4005 |

以 `typhoon/evaluate_preference_monotonicity.py` 沿三個 edge（count→gt、count→risk、
risk→count；第三軸固定 0.15）以 step 0.25 掃描 60 個 paired held-out cases，Phase 9B
checkpoint 的違規率：

| 指標 | rising_gt edge | 三 edge 整體 |
|---|---:|---:|
| 相鄰非單調率 | 17.1% | 15.4% |
| 端點反轉率 | 28.3% (17/60) | 28.3% |

`monotonicity_gate`（兩者皆 < 5%）= **false**。

## 根因假設（依懷疑度排序）

1. **experts 在 GT／risk 軸分化不足**：四個 centroid 的撤離 GT 只落在 291k–308k（約 5%
   spread）。soft router 只能混合 experts 已編碼的行為。
2. **argmax 解碼的離散不連續**：混合後對 masked 機率取 deterministic argmax，router 權重的
   連續微移會翻轉單一 dispatch；每次 dispatch 是離散 GT 跳躍，KPI 因而非單調。
3. **router 欠訓練 / anchor CE 過強**：把 router 釘在四個 centroid，中間偏好靠內插。

## 修復 1：tie-aware 解碼（已實作，經測不足）

deterministic 解碼時，對混合機率落在最大值 `decode_tie_eps` 內的並列動作，改以
preference-weighted 單步 GT/risk 分數破並列。核心洞察：observation 內每艘船的 `log1p(GT)`
與 `risk/5` 特徵，對同狀態下的 `GT_i/GT_total`、`risk_i/risk_total` 嚴格同序，因此
`w_gt·log_gt + w_risk·risk` 為並列船排序，等同 preference-weighted 單步目標排序（count 的
`1/N` 對船間無區辨，略去）；wait 動作給固定低分，使等機率 dispatch 在並列時優先。預設
`decode_tie_eps = 0.0`（純 argmax，byte-identical），opt-in 並存入 checkpoint metadata。

**實測（seed-42，2026-07-25）**：掃描 `decode_tie_eps ∈ {0, 0.001…0.05}`，端點反轉只從
17→16/60，相鄰非單調率不降反微升，`eps ≥ 0.02` 後 count utility 開始退步。decode-step 診斷
給出原因：在 1,919 個「有 ≥2 艘可行船」的真正選擇點上，top1–top2 機率 gap 中位數 **0.61**，
`eps=0.01` 內的近似並列只佔 **1.3%**。這是 **confident-but-non-monotone routing**，不是 tie
問題。tie-aware 解碼因此保留為正確、便宜、安全的第一道防線（experts 分化改善或狀態真正並列
時才發揮），但不是本 checkpoint 的解藥。

## 修復 2：router 單調性正則（本 phase 主要 lever）

experts 凍結後，偏好只透過 routing weights 影響混合分布，因此可在 argmax **之前** 的可微
機率上直接施加單調性約束。

令凍結 expert `k` 在觀測 `s` 下對動作 `a` 的機率為 `p_k(a|s)`（不含偏好、無梯度）。以
observation 的每艘船特徵建立可微、且與真實目標同序的單步目標 proxy：

- `O_count(a) = 1`（dispatch）／`0`（wait）；
- `O_gt(a) = log1p(GT_a)/log1p(250k)`（即 feature 0）；
- `O_risk(a) = risk_a/5`（即 feature 3）。

expert `k` 的期望單步目標 `m_{k,j}(s) = Σ_a p_k(a|s)·O_j(a)`（detach，無梯度）。混合後在偏好
`w` 下的期望目標為 `E_j(s,w) = Σ_k r_k(s,w)·m_{k,j}(s)`，其中 `r_k = softmax(router_logits)`
可微。

每個 minibatch，對每個目標 `j`：把該 transition 的偏好在 `j` 軸加 `delta` 後重正規化得
`w'`（`w'_j = (w_j+delta)/(1+delta) > w_j`），以 pairwise hinge 懲罰

```
L_mono = Σ_j  mean( relu( E_j(s, w) − E_j(s, w') ) )
```

即「提高 j 權重卻降低期望 j 目標」時才受罰。只更新 router 與 critic；experts、安全 mask、
Phase 9B 的 preference-first scalar advantage 與 anchor CE 全部保留。總 loss 加
`monotonicity_coef · L_mono`。這是對全域單調性的 **局部代理**：若每個狀態下提高 j 權重都
提高期望 j 目標，軌跡層級的累積 j 目標亦傾向單調上升；不保證 0 違規，但直接壓低違規率。

### 超參數與執行

以 Phase 9B（distilled Phase 9A）為起點，`--monotonicity-coef 0.5`、`--monotonicity-delta
0.1`，其餘沿用 Phase 9B pilot（25,000 steps/profile、100k transitions、router/critic LR
`3e-4`、anchor `0.01`）。

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.train_soft_moe_ppo \
  --db /path/to/ua1008l.sqlite --seed 42 --steps-per-profile 25000 \
  --output-dir typhoon/models/soft_moe_router_monotone \
  --monotonicity-coef 0.5 --monotonicity-delta 0.1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.evaluate_preference_monotonicity \
  --db /path/to/ua1008l.sqlite --step 0.25 \
  --model typhoon/models/soft_moe_router_monotone/seed-42/final.pt

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.evaluate_soft_moe_grid \
  --db /path/to/ua1008l.sqlite --step 0.25 \
  --after-model typhoon/models/soft_moe_router_monotone/seed-42/final.pt
~~~

## 驗收標準

- 相鄰非單調率 < 5%、端點反轉率 < 5%（`monotonicity_gate=true`）；
- GT/risk cross-utility 對角最佳維持 2/2；
- safety violations、rejected actions 均為 0；
- 0.25 連續 grid 平均 utility 相對 Phase 9B 不退步。

## 結果（seed-42）

三種 lever 都試過，**deterministic-trajectory 單調性沒有改善**（負結果，但把根因收斂得很清楚）：

| 方法 | 相鄰非單調率 | 端點反轉率 | GT/risk gate | safety | grid 平均 delta |
|---|---:|---:|---:|---:|---:|
| Phase 9B baseline | 15.4% | 28.3% | 2/2 | 0 | — |
| + tie-aware 解碼 | 15.7% | 28.3% (16/60) | 2/2 | 0 | — |
| + router 單調性正則（hinge, coef 0.5） | 15.3% | 28.3% | 2/2 | 0 | +0.000578 |
| + router 單調性正則（ranking, coef 1.0, anchor 0.002） | 16.1% | 28.3% | 2/2 | 0 | +0.000477 |

`monotonicity_gate` 全部 = false。GT/risk 對角最佳 2/2、safety 0、grid 平均 utility 不退步
（未改善也未惡化），四個 centroid KPI 不變。

### 為什麼 router 正則沒用（已量化）

hinge 版的 mean monotonicity_loss 僅 0.0014、ranking 版 0.019——正則確實在作用，但改不動
deterministic KPI。關鍵診斷：在 251 個 held-out 多選擇狀態上，**期望**單步 GT 的相鄰非單調
率高達 **52.3%**，但幅度極小。也就是說在中間偏好，routing weights 對偏好幾乎沒反應（被
anchor 與蒸餾初始化釘住），期望目標近乎平坦、方向一半錯但量級微弱，因此：

1. hinge 的梯度隨量級消失（magnitude-dominated）；
2. 改用非零梯度的 ranking loss、並把 anchor 降到 0.002，期望目標仍被 frozen experts 的相似度
   壓住無法拉開。

更根本的是：**期望目標（連續）與 deterministic argmax 軌跡（離散）解耦**。即使把期望目標推成
單調，argmax 每步只挑一艘船、下游整段軌跡隨之改變，累積 KPI 仍非單調。這把根因鎖定在
假設 1（experts 分化不足，evac_GT spread 僅 5%）＋假設 2（argmax 離散不連續），而非 router
對偏好的平滑反應。

### 結論與交棒

router-level 正則與 decode-level tie-break 都無法修復此 checkpoint 的軌跡單調性。真正的
lever 落在 **修復 3：重蒸餾／分離四個 experts**（加 inter-expert separation 目標，讓 GT／risk
expert 在對應軸真正拉開），或改用 preference-aware 的軌跡層級解碼/搜尋——兩者都屬 Phase 11
的 expert 重訓範疇，不在 frozen-router 範圍內。本 phase 交付診斷工具與兩個已驗證無效的
router lever，並精確定位根因；Phase 9B 仍為前端「實驗性連續偏好」的預設，Phase 10 checkpoint
僅為診斷產物，未接受為前端模型（未通過 monotonicity gate）。

## 交付

- `soft_moe.py`：`decode_tie_eps` + tie-aware 解碼、`_action_objective_proxy`、
  `expert_expected_objectives`。
- `train_soft_moe_ppo.py`：`router_monotonicity_loss` 與 `--monotonicity-coef` /
  `--monotonicity-delta`（預設 0 = Phase 9B 行為）。
- `evaluate_preference_monotonicity.py`：三 edge 單調性診斷與 gate。
- `tests/test_soft_moe.py`、`tests/test_soft_moe_ppo.py`：解碼與正則的單元測試。
