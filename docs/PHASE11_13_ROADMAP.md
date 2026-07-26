# Phase 11–13 規劃：競爭力、驗收 gate 與服務化

本文件承接 Phase 10（偏好單調性，見 `PHASE10_PREFERENCE_MONOTONICITY.md`），規劃三個尚未
開始的 phase：(11) 對規則基線的競爭力、(12) 多 seed／細 grid／CI 的正式驗收 gate、
(13) 服務化與數位孿生整合契約。各 phase 以下方實測 baseline 為量化依據。

## 共同實測 baseline（Phase 9B checkpoint, seed-42, held-out）

RL 對規則基線的配對 utility（同 case、同安全限制）：

| 偏好 | RL | FCFS | Risk-aware | Value-density |
|---|---:|---:|---:|---:|
| count | 0.4018 | 0.3990 | 0.3722 | **0.4453** |
| balanced | 0.3813 | 0.3814 | 0.3776 | **0.4184** |
| gt | **0.3669** | 0.3575 | 0.3851 | 0.3595 |
| risk | 0.4005 | 0.3878 | 0.3755 | **0.4505** |

value-density 在 count／balanced／risk 明顯領先 RL，RL 只有在 GT 偏好上勝過 value-density
（但輸給 risk-aware）。cross-utility 對角最佳維持 2/4（gt、risk）。

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

## Phase 12：多 seed、細 grid 與 CI（正式驗收 gate）

把目前的下一步正式化為 deployment 前的硬 gate：

- 訓練 seed 43、44（建議再加 45、46）；
- grid 細化到 0.1；
- 每點加 paired bootstrap 95% CI 與 per-point regret，對照 value-density、對應 specialist 與
  Phase 11 oracle；
- experts 維持凍結，直到多 seed 同時通過 Phase 10 單調性 gate、GT/risk 2/2 gate 與非負平均
  delta 才考慮延長 steps 或 joint fine-tune。
- 建議訊息：`feat: multi-seed continuous evaluation with bootstrap CI`

## Phase 13：服務化／數位孿生整合契約

回答「若要當成服務（例如數位孿生會用到這個 RL 排程），需要被設定什麼」。目前 API 是一次性
重播整段 episode，情境靠 `build_scenario.py` 由歷史 DB 合成；服務化需要把「消費端必須提供
什麼」變成明確、可驗證的契約。

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

### 需新增的介面與產物

- **ServiceTimeSource adapter**：過航／靠泊／離泊時間一律經可注入的 provider（鐵律）；文件化
  孿生需實作的介面，模擬器內不得寫死在泊時間。
- **capability／limits manifest（機器可讀，擴充 model_cards 並由 `/health` 輸出）**：固定
  30 艘 obs 契約、合法偏好範圍、哪些欄位是合成 vs 實測、2/4 對角與 GT-risk gate、安全保證
  （mask 強制、0 違規）、單 seed pilot 的 caveat。讓孿生清楚知道能宣稱與不能宣稱什麼。
- **stateful stepping API**：現有 API 一次跑完整 episode；孿生需要 `POST /step`（推進到下一
  事件、依當前 live 狀態回傳下一步 dispatch／wait 建議），並在新 ETA／資源更新時重排。契約
  須維持 safety mask；模型缺失時維持 503、不做 heuristic fallback（既有 invariant）。
- **可變船數**：折入 `WORKFLOW.md` 末「30 艘容量與部署待辦」的 rolling top-30 window、空 slot
  永久 mask 與 KPI 分母只計真實船舶；此路徑需以不同實際船數重訓與重評，不能直接沿用現有
  checkpoint。

### 驗收

- config schema 有測試驗證；`ServiceTimeSource` 可替換；capability manifest 由 `/health`
  輸出；可變船數路徑以自身重訓為 gate；安全 invariant 不變。
- 建議訊息：`feat: define digital-twin scenario and service contract`
