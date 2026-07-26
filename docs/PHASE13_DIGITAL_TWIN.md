# Phase 13：RL Pareto 數位孿生整合

## 服務目標

數位孿生取得的是 **RL policy 產生的多偏好 Pareto 排程集合**。Value-density、FCFS 與
Risk-aware 僅存在研究比較端點與離線評估，不會被混入 `pareto_rl`，也不會在 RL checkpoint
缺失時作為靜默 fallback。

數位孿生端點使用最新的 reward-trained Phase 10 Soft MoE checkpoint。count、balanced、GT、
risk 四個 centroid，以及最多 12 組選配連續三維偏好，全部由**同一顆模型**推理。

這是第一版 PoC 契約，不代表 Phase 10 已通過正式部署 gate。回應會明確帶出 candidate 的
`policy_source`、checkpoint family 與目前能力限制。

## 整合模式

第一版使用 **stateless receding horizon**：每次請求代表一個完整決策時點。當船舶 ETA、拖船
容量、封港時間或風險資料改變時，數位孿生重新送入 snapshot；服務不在伺服器保存港區真實狀態。

~~~text
數位孿生 snapshot
  ├─ 30 艘待撤船舶
  ├─ observed_at / closure_at
  ├─ 拖船與入口容量
  └─ 選配連續偏好
          │
          ▼
ServiceTimeSource → TyphoonEvacuationEnv → RL policies
                                             │
                    ┌────────────────────────┴──────────────────────┐
                    ▼                                               ▼
          完整 RL candidates                              非支配 pareto_rl
                                                                    │
                                                                    ▼
                                                        各候選下一步動作
~~~

## 輸入契約

`DigitalTwinScenarioRequest` 必須包含：

| 欄位 | 第一版規則 |
|---|---|
| `request_id` | 呼叫端可追蹤的唯一請求識別 |
| `observed_at` | 含 timezone offset 的決策時間 |
| `closure_at` | 含 timezone offset；距決策時間 2–12 小時 |
| `resources.tug_capacity` | 2–12 |
| `resources.entrance_capacity` | 目前固定 `{ "1": 1, "2": 1 }` |
| `vessels` | 恰好 30 艘，`ship_id` 不可重複 |
| 船舶必要欄位 | GT、入口、`ready_at`、`transit_hours`、拖船需求、風險點 |
| `continuous_preferences` | 選填，最多 12 組；每組 `(count, GT, risk)` 非負且總和大於 0 |

`transit_hours` 不由 API 內部 GT 查表產生。預設 `RequestServiceTimeSource` 直接使用數位孿生
提供的值；整合端也能在 Python 內注入其他 `ServiceTimeSource` provider。若 service time 超出
目前約 0.36–1.65 小時的訓練分布，請求仍可執行，但回應會包含 OOD warning。

## API

### 能力宣告

~~~http
GET /api/typhoon/digital-twin/capabilities
~~~

回傳 schema 版本、固定船數、時間範圍、RL candidate 類型、安全 mask 與尚未支援的能力。
同一份 manifest 也包含在 `/api/typhoon/health`。

`GET /api/typhoon/digital-twin/health` 會另外實際載入最新 Phase 10 checkpoint，檢查 observation
為 344、action 為 31，並回傳 model family、SHA-256 與 validation record。此端點不依賴歷史 DB 或
Phase 8 experts，應作為數位孿生整合的 readiness probe。

### 完整 Pareto 排程

~~~http
POST /api/typhoon/digital-twin/pareto
Content-Type: application/json
~~~

主要回應欄位：

- `rl_candidates`：所有實際執行的 RL 排程；
- `pareto_rl`：以撤離艘數、GT、風險點去重後的非支配集合；
- `candidate_id`／`policy_source`：候選使用的偏好與 Phase 10 checkpoint family；
- `schedule`／`decisions`／`kpi`：完整時程、動作 trace 與預估 KPI；
- `warnings`／`capability_limits`：分布外輸入與已知限制。

### 下一步建議

~~~http
POST /api/typhoon/digital-twin/step
Content-Type: application/json
~~~

輸入與 `/pareto` 相同。輸出只保留每個非支配候選的第一個 `dispatch` 或 `wait`，以及該完整
排程的 projected KPI。孿生執行或否決動作後，應以新的 `observed_at` 和船舶／資源狀態重送。

FastAPI 會在 `/docs` 產生完整 OpenAPI schema，包含 30 艘 validation 與所有巢狀欄位。

## 現在不能宣稱的能力

- 不支援少於或多於 30 艘；top-30／padding adapter 尚未重訓驗收；
- 不支援請求內的 active operations 與未來拖船／入口釋放事件；
- `/step` 是 snapshot 的 next-action projection，不是 server-side stateful session；
- Phase 10 只有 seed 42，且 preference monotonicity gate 未通過；
- 現有風險點若仍使用 synthetic proxy，不得描述為港方正式風險順位；
- RL 尚未追平 VD／oracle；VD 只作 benchmark，不能把規則結果標示為 RL。

## 下一個實作階段

1. 加入 `active_operations` 與各資源 release time，讓 snapshot 能表示港區進行中作業；
2. 將環境狀態初始化抽成正式 adapter，支援孿生事件後的精確續排；
3. 實作 rolling top-30 與 permanent padding mask，重新訓練可變船數模型；
4. 接入真實 service-time provider，建立貪婪法不再近最佳的非短視訓練情境；
5. 對 RL Pareto 模型執行多 seed、0.1 preference grid、bootstrap CI、hypervolume、epsilon 與
   oracle／VD regret gate。
