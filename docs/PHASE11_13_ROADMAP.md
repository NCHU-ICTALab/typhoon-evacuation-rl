# Phase 11–13 規劃：競爭力、驗收 gate 與服務化

本文件承接 Phase 10（偏好單調性，見 `PHASE10_PREFERENCE_MONOTONICITY.md`）。Phase 11 的
oracle 與 Phase 12 的 VD 蒸餾診斷已完成；Phase 13 的服務化／數位孿生整合已開始。正式服務
目標是 **RL 產生的多偏好 Pareto 集**，VD 僅作 teacher／benchmark，不是主策略。

## 共同實測 baseline（Phase 9B checkpoint, seed-42, held-out）

Phase 11 以正確的 per-preference VD 口徑重算後，Phase 9B RL 四種偏好均落後 VD 約 10–13%；
VD 距 oracle 約 1–3%。完整數字以 [PHASE11_COMPETITIVENESS.md](PHASE11_COMPETITIVENESS.md)
與 `model_cards/phase11_oracle_seed42.json` 為準。這是 RL 的品質基準，不改變 RL-only 服務契約。

## Phase 11：對規則基線的競爭力

### 問題定義

value-density 是 preference-aware 貪婪 heuristic（依 `dispatch_value / transit_hour` 選船，
不用洩漏資訊），在多數偏好上領先 RL 約 +10%。目前 utility 幾乎可分離
（`count/N + GT/total + risk/total`），耦合只來自拖船、入口與封港硬限制，這種近似 knapsack
的問題本來就對貪婪有利。

> **更新（已完成）**：步驟一的嚴格 oracle 已實作並定案，見
> [PHASE11_COMPETITIVENESS.md](PHASE11_COMPETITIVENESS.md)。結論：value-density 近全域最佳
> （regret 1–3%），RL 落後 VD 約 10–13%。因此走「VD ≈ oracle」分支——目標改為在 CI 內追平
> VD（步驟二的 warm-start／多 seed），不追求勝過。以下保留原始規劃脈絡。

### 步驟一：先量出最佳化上限（決定 RL 是否有空間）

deterministic 情境（`randomize=False`、固定 jitter seed）是完全已知的有限排程問題。對每個
case 以 beam search／ILP／DP 在相同 hard mask 下最大化 scalarized utility，求 per-case oracle：

- **VD ≈ oracle**（count/balanced/risk）：這些偏好接近可分離、貪婪即近最佳，RL 不應被要求
  「贏過」VD。正確口徑改為「RL 在 CI 內與 VD 打平，同時提供 VD 沒有的連續偏好控制與非支配
  Pareto 集」——以 hypervolume／epsilon 證明 RL 的 Pareto 集具競爭力。
- **oracle ≫ VD**（封港緊迫下的排序陷阱）：有真實 headroom，值得往下改 RL。

### 步驟二：若有 headroom 才動 RL

1. **以 VD 當 advantage baseline / warm-start**：先用 VD 行為克隆 experts，再 PPO 微調，讓
   模型只在「貪婪短視會錯」的緊迫封港排序上學到超越（learn to beat greedy）。
2. **低 LR joint fine-tune**：解凍 experts、以小 learning rate 聯合微調（須先通過多 seed
   gate）。
3. **課程／取樣**：oversample 封港緊、資源少、貪婪失效的 episode。
4. 補上目前缺的 **GT-aware 基線**，讓 GT 偏好有對應的規則對手。
5. **experts 重蒸餾／分離（承接 Phase 10 負結果）**：Phase 10 證實 frozen-router 的
   單調性正則與 tie-aware 解碼都無法修復連續偏好下的 GT 排序異常，根因是四個 experts 分化
   不足（evac_GT spread 僅 5%）＋argmax 離散不連續。在重蒸餾時加 inter-expert separation
   目標（例如懲罰 GT／risk expert 在對應軸的期望目標重疊），讓 argmax 隨偏好自然單調；重訓
   後再跑 `evaluate_preference_monotonicity` 驗收 gate < 5%。詳見
   [PHASE10_PREFERENCE_MONOTONICITY.md](PHASE10_PREFERENCE_MONOTONICITY.md)。

### 驗收

- 至少在 GT 以外再有一種偏好，paired RL utility ≥ 最佳規則 − CI；或在緊迫封港子集上 RL
  非支配集支配規則集。報告 hypervolume、epsilon indicator 與對 oracle／VD 的 regret。
- 建議訊息：`feat: add optimization oracle and greedy-competitive training`

## Phase 12：VD 蒸餾診斷（已完成）

BC 與 DAgger 已驗證 97–98% per-step 模仿率仍會因 closed-loop covariate shift 落後 VD 約
9–13%。此 checkpoint 不接受為正式 RL 主模型；VD 保留為 teacher／benchmark。詳見
[PHASE12_VD_DISTILLATION.md](PHASE12_VD_DISTILLATION.md)。

多 seed、0.1 grid、paired bootstrap 95% CI、per-point oracle／VD regret 仍是部署硬 gate，
但改列為 RL Pareto 模型的跨 phase 驗收工作，不再與 Phase 12 編號衝突。

## Phase 13：服務化／數位孿生整合契約

回答「若數位孿生要使用 RL 多 Pareto 排程，必須提供什麼」。第一版已採 stateless
receding-horizon snapshot：孿生在 ETA、資源或封港時間更新時重新送入完整快照，服務回傳完整
RL Pareto 集，或每個 Pareto 候選的下一步動作。

### 消費端必須提供的設定（提議 `EvacuationScenarioConfig`，Pydantic 型別化）

1. **fleet**：船舶清單，每艘含穩定 id、`gross_tonnage`、`entrance∈{1,2}`、拖船需求、
   `ready_hour`（由首報 ETA 衍生的整備時間——遵守 first_captain_eta 洩漏鐵律）、
   `transit_hour`（必須經 `ServiceTimeSource` 提供，不可寫死——CLAUDE.md 鐵律）、風險分量
   （hazardous／mobility／draft／engine proxy，或外部官方風險向量）。
2. **resources**：`tug_capacity`、每個入口的 `entrance_capacity`。
3. **closure**：每個入口的 `closure_hour`（由 `rules.py` 的風力硬限制套在數位孿生的氣象
   feed 上產生）——不可交換的硬截止。
4. **clock**：絕對時間 `now`；孿生推進時間並在事件發生時要求重排。
5. **preference**：`(count, GT, risk)` 三維非負權重。

### 已加入的第一版介面與產物

- `typhoon/digital_twin.py`：Pydantic snapshot schema、可注入 `ServiceTimeSource`、能力限制與
  scenario adapter。
- `GET /api/typhoon/digital-twin/capabilities`：固定 30 艘、合法時間／入口／資源範圍與不支援
  項目的機器可讀宣告。
- `POST /api/typhoon/digital-twin/pareto`：以同一顆 Phase 10 Soft MoE 跑四個 centroid 加選配
  continuous preference，回傳全部 RL candidates 與非支配 `pareto_rl`；不執行規則候選。
- `POST /api/typhoon/digital-twin/step`：從同一 snapshot 投影每個 Pareto 候選的第一個
  dispatch／wait，供孿生以 receding horizon 執行後重排。
- 模型缺失時維持 503，不做 heuristic fallback。
- **可變船數**：折入 `WORKFLOW.md` 末「30 艘容量與部署待辦」的 rolling top-30 window、空 slot
  永久 mask 與 KPI 分母只計真實船舶；此路徑需以不同實際船數重訓與重評，不能直接沿用現有
  checkpoint。

### 第一版驗收與後續

- schema、`ServiceTimeSource` 替換、RL-only candidate contract、capabilities 與 `/step` 已有測試；
- active operations 尚未支援，第一版不是 server-side stateful session；
- 可變船數、active resource release、真實 service time、多 seed gate 完成後，才能升級正式服務；
- 詳細契約見 [PHASE13_DIGITAL_TWIN.md](PHASE13_DIGITAL_TWIN.md)。
- 建議訊息：`feat: add RL Pareto digital-twin snapshot API`
