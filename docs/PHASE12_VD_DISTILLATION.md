# Phase 12：VD 蒸餾為可服務 checkpoint（BC / DAgger）與定案

## 目標

打造一個**可服務的單一 checkpoint**：乾淨追平 value-density(VD)、保有連續偏好控制、安全
gate 與 Pareto 集，且**不覆蓋現有 checkpoint**（現有 Phase 9B 供 Pareto 演示）。

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

## 定案

**在目前近可分離的合成環境，任何神經策略（RL／BC／DAgger）都無法乾淨追平 VD**：
- reward-driven RL：regret 11–16%（Phase 9B）；
- BC of VD：covariate shift，差 ~10%；
- DAgger of VD：仍差 ~9–13%。

而 **value-density 本身已同時滿足目標的四個條件**——可服務（確定性規則，API 已將其作為
baseline 計算）、連續偏好（`偏好·objectives/transit` 對任意三維偏好都成立）、安全（只從 action
mask 選）、Pareto（51/60 distinct）、外加近最佳（距上限 1–3%）與單調（0/60）。

**因此本目標的正解是「直接服務 value-density」**——把 VD 放在偏好／安全／Pareto 契約後面當
主策略，而不是硬塞一個在每個面向都更差的神經近似。neural 路線（BC-init + PPO fine-tune）只有
在環境獲得**非短視結構**（真實靠泊／離泊 service time，`ServiceTimeSource`，資料源尚未到位）、
使貪婪不再近最佳時，才值得重啟；屆時本 phase 的 BC/DAgger pipeline 可當 warm-start 骨架。

## 交付

- `train_behavior_clone.py`、`train_dagger.py`（VD 蒸餾骨架，供未來非短視環境的 warm-start）；
- checkpoint 寫入新目錄，未覆蓋任何現有模型；
- 定案：對「可服務、追平 VD」目標，服務 VD 本身為最佳解；neural checkpoint 延後至 service-time
  資料到位。
