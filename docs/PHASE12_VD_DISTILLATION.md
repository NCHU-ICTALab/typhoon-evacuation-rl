# Phase 12：VD 蒸餾診斷（BC / DAgger）

## 目標

檢驗是否能把 value-density(VD) 的近最佳行為蒸餾進單一神經 checkpoint，同時保有連續偏好
控制、安全 gate 與 Pareto 集，且**不覆蓋現有 checkpoint**（現有 Phase 9B 供 Pareto 演示）。
VD 在此是 teacher／競爭力基準，不是預設服務策略。

## 方法

Phase 11 已證 VD 近全域最佳（regret 1–3%）、且偏好單調（0/60 端點反轉）、又能差異化
（51/60 distinct schedules）。因此策略是把 VD 蒸餾進 soft-MoE：四個 expert 各自在其 centroid
模仿 VD（masked cross-entropy behavior cloning），再蒸餾 router；全部寫入**新目錄**
（`models/vd_bc_experts/`、`models/soft_moe_vd_bc/`、`models/soft_moe_vd_dagger/`）。
covariate shift 以 DAgger 修正（rollout 學生、以 VD 標記其造訪狀態、聚合重訓）。

- `typhoon/train_behavior_clone.py`：VD → 四 expert 的 BC + router 蒸餾。
- `typhoon/train_dagger.py`：DAgger 迴圈（teacher + student 狀態聚合）。

## 結果（seed-42, 60 held-out cases × 4 centroids）

| 方法 | 對 VD 的 utility 差距 | 端點反轉 | per-step 模仿正確率 | 安全 |
|---|---:|---:|---:|---:|
| Phase 9B RL（reward PPO） | −0.042…−0.057 | 17/60 | — | 0 |
| BC of VD | −0.036…−0.052 | 30/60 | 97–98% | 0 |
| DAgger of VD（2 rounds） | −0.034…−0.056 | 18/60 | 97–98% | 0 |
| **value-density 本身** | **0（基準）** | **0/60** | — | 0 |

診斷：standalone BC expert 與 soft-MoE 混合輸出**完全相同**（例：count 都是 0.4012），證明
router／混合是乾淨的，差距全在 expert 的模仿本身。即使 per-step 模仿正確率 97–98%，closed-loop
utility 仍落後 VD ~10%——這是典型 **behavior-cloning covariate shift**：~3% 的每步誤差落在封港
緊迫的關鍵派船決策上並沿軌跡放大。DAgger（以 VD 標記學生造訪狀態）只把差距縮小一點點，
仍差 ~9–13%，且單調性未回到 VD 水準。

## 研究結論與服務目標更正

**在目前近可分離的合成環境，任何神經策略（RL／BC／DAgger）都無法乾淨追平 VD**：
- reward-driven RL：regret 11–16%（Phase 9B）；
- BC of VD：covariate shift，差 ~10%；
- DAgger of VD：仍差 ~9–13%。

而 **value-density 本身已同時滿足目標的四個條件**——可服務（確定性規則，API 已將其作為
baseline 計算）、連續偏好（`偏好·objectives/transit` 對任意三維偏好都成立）、安全（只從 action
mask 選）、Pareto（51/60 distinct）、外加近最佳（距上限 1–3%）與單調（0/60）。

原始實驗曾據此建議直接服務 VD；**專案服務目標現已明確更正為 RL 多偏好／多 Pareto 集**，
因此不採用該建議。正確解讀是：本 phase 證明目前 BC／DAgger checkpoint 還不能作為正式 RL
主模型，而不是證明應把規則方法改名成 AI 或混入 RL 候選。

後續做法是：

1. Phase 8／9B 繼續作為有明確限制標示的 RL Pareto PoC；
2. VD 只用於 offline teacher、regret 基準與訓練課程，不出現在 RL-only 服務候選；
3. Phase 13 先完成數位孿生契約與 receding-horizon RL 服務；
4. 以真實 `ServiceTimeSource` 加入非短視結構，再進行 BC warm-start + preference-conditioned
   PPO／trajectory-level search，重新驗收 utility、單調性、Pareto coverage 與多 seed CI。

## 交付

- `train_behavior_clone.py`、`train_dagger.py`（VD 蒸餾骨架，供未來非短視環境的 warm-start）；
- checkpoint 寫入新目錄，未覆蓋任何現有模型；
- 負結果定案：現有 BC／DAgger 不接受為 RL 主模型；VD 保留為 teacher／benchmark。
- 服務定案：正式候選必須來自 RL policy；規則基線不得混入 `pareto_rl`。
