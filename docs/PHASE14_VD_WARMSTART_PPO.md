# Phase 14：VD warm-start preference-conditioned PPO

## 目的

Phase 12 已把 preference-aware value-density teacher 蒸餾成單一 Soft MoE，但 BC／DAgger 的
少量關鍵錯誤會沿 closed-loop 放大。只再訓練 router 無法修正 expert 本身，因此本階段使用：

```text
VD teacher → BC experts → DAgger experts
                         │
                         ▼
          fixed-preference reward PPO（四 experts）
                         │
                         ▼
          continuous-preference router PPO
                         │
                         ▼
              新 Soft MoE checkpoint
```

這仍是同一顆 checkpoint 的多偏好服務架構。VD 只提供初始化 lineage 與離線 benchmark，推理
時不會執行 VD，也不會把規則候選混入 `pareto_rl`。

## 為何分兩段

Phase 12 的 standalone expert 與 Soft MoE mixture 已驗證一致，表示主要差距不是 router wiring，
而是 expert imitation error。第一段用各 centroid 的真實 vector reward、preference-first scalar
advantage、單次 normalization／PPO clipping 與 preference-scalar critic fine-tune expert。
成功的 25k checkpoint 使用原始 DAgger logits（`--exploration-temperature 1`），且不重設或
detach critic。temperature=3、value-head reset 與 detached critic 保留為診斷旗標；它們不改初始
deterministic action ordering，但 seed-42 pilots 的 GT 退步較大，因此不是正式候選預設。
第二段才凍結更新後的 experts，沿用 Phase 12 preference/state router，使用 centroid anchor 與
Dirichlet continuous preferences 做 router PPO。

## 執行

Pilot 不會被接受為服務 checkpoint：

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python \
  -m typhoon.train_vd_warmstart_ppo \
  --db ../sdci_data/ua1008l.sqlite \
  --expert-steps 2000 \
  --router-steps-per-profile 2000 \
  --exploration-temperature 1 \
  --epochs 3 \
  --pilot
```

正式候選可提高到每 expert／profile 25,000 steps，且至少重跑 seeds 42、43、44。

## Service gate

四對角只保留為診斷，不是硬 gate；count／balanced 在既有 experts 本來就不一定對角最佳。
Phase 14 以現行服務的 Phase 10 為 reference：

- pilot 必定拒絕；
- safety violations 與 rejected actions 都為 0；
- 至少 80% paired cases 因偏好改變 action trace；
- 偏好差異 cases 相較 Phase 10 最多減少 2；
- 四個 centroid utility 平均改善，且至少三個 profiles 改善；
- 任一 profile utility 相較 Phase 10 不得退步超過 `0.002`；
- 平均 specialization margin 相較 Phase 10 不得退步超過 `0.001`。

VD／oracle regret 繼續保留為模型品質研究指標，但不再用來否決「比 Phase 10 更適合目前
RL-only 多 Pareto PoC」的 checkpoint。

即使單 seed 通過，也只能列為候選；正式替換 Phase 10 前仍須 continuous 0.1 grid、偏好單調性、
bootstrap CI、多 seed 與數位孿生真實 service-time OOD 驗收。

## Seed 42 實測（held-out 60 paired cases）

VD teacher 本身的四列皆為對角最佳，但這只證明 teacher 行為分離，不是現行服務 gate。
Phase 10 的比較基準為 56/60 cases 產生偏好差異；Phase 14 應比較 centroid utility、
specialization margin 與差異覆蓋，而不是要求 4/4 對角。

| Fine-tune | Steps（expert / router profile） | Phase 10 utility 平均 delta | 最差 profile delta | 差異 cases | 新 gate |
|---|---:|---:|---:|---:|---|
| 原始 PPO | 25k / 25k | +0.003975 | -0.0019（GT） | 55/60 | **通過** |
| temperature=3 | 10k / 10k | +0.003075 | -0.0046（GT） | 55/60 | pilot／GT 退步，拒絕 |
| temperature=3 + neutral/detached critic | 10k / 10k | +0.002100 | -0.0026（GT） | 54/60 | pilot／GT 退步，拒絕 |

三輪皆維持 safety violations 與 rejected actions 為 0。原始 25k checkpoint 相對 Phase 10
改善 count、balanced、risk，GT 僅退步 `0.0019`，平均 utility 改善 `0.003975`；差異覆蓋與
specialization margin 亦在容忍範圍，因此通過目前 RL-only 多 Pareto PoC 的替換 gate。
兩個 10k pilots 的 GT 退步超過 `0.002`，不採用。

下一輪不應再盲目延長相同 PPO。建議使用 teacher-regularized PPO：在 on-policy student states 同時
取得 VD teacher action，以 PPO reward loss 為主、teacher cross-entropy／KL 為保護項，並按
held-out profile utility 選擇各 expert checkpoint。這仍是 PPO fine-tune，推理時仍只執行單一
Soft MoE，不會呼叫 VD；teacher 只存在於訓練階段。

## 輸出

所有輸出寫入 `models/soft_moe_vd_warmstart_ppo/seed-<seed>/`，不覆蓋 Phase 10 或 Phase 12：

- `experts/<profile>/final.pt`
- `final.pt`
- `training_manifest.json`
- `evaluation.json`
- `comparison.json`
- `phase10_service_assessment.json`
- `service_candidate.pt`（相同權重，metadata 使用修正後的 Phase 10 相對 gate）

目前實驗輸出：

- `models/soft_moe_vd_warmstart_ppo/seed-42/`：原始 25k 正式 run，通過 Phase 10 相對 PoC gate；
- `models/soft_moe_vd_warmstart_ppo_t3_pilot/seed-42/`：temperature pilot；
- `models/soft_moe_vd_warmstart_ppo_t3_detached_pilot/seed-42/`：critic 隔離 pilot。
