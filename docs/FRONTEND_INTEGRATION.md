# Phase 14 數位孿生前端整合

## 目標

前端或數位孿生每個決策時點送出一份港區 snapshot。Phase 14 preference-conditioned PPO
以相同 checkpoint 對四個固定偏好及選填的連續偏好進行推論，回傳 RL-only Pareto 排程。

服務不發布封港命令，也不直接控制船舶。數位孿生負責顯示泊位、航道與船舶座標；RL
只提供派船／等待決策與相對時間。

## 啟動

需要 Python 3.11 以上版本；第一次安裝相依套件時需要網路。

~~~bash
git clone https://github.com/NCHU-ICTALab/typhoon-evacuation-rl.git
cd typhoon-evacuation-rl
./scripts/run_demo_backend.sh
~~~

預設監聽 `0.0.0.0:8765`。可使用下列環境變數：

| 環境變數 | 預設 | 用途 |
|---|---|---|
| `TYPHOON_HOST` | `0.0.0.0` | 後端監聽位址 |
| `TYPHOON_PORT` | `8765` | 後端連接埠 |
| `TYPHOON_CORS_ORIGINS` | `*` | 逗號分隔的允許前端 origins；正式部署應限制 |
| `TYPHOON_PHASE14_MODEL_PATH` | repo 內 service candidate | 選填的 checkpoint 覆寫路徑；仍需通過固定 SHA |

此推論路徑不讀取 `ua1008l.sqlite`。repo 內只發布 Phase 14 service candidate，啟動載入時會
驗證 SHA-256；歷史資料與其他訓練 checkpoint 不在交付範圍。

## API

| 方法 | 路徑 | 用途 |
|---|---|---|
| `GET` | `/api/typhoon/health` | Phase 14 readiness、模型 family、SHA 與 shape |
| `GET` | `/api/typhoon/digital-twin/capabilities` | 機器可讀的契約與限制 |
| `POST` | `/api/typhoon/digital-twin/pareto` | 回傳所有 RL 候選及非支配 Pareto 集 |
| `POST` | `/api/typhoon/digital-twin/step` | 回傳每個 Pareto 候選的第一個動作 |
| `GET` | `/docs` | FastAPI OpenAPI |

隊友首次整合建議先使用 `/step`，因為回傳較小，適合數位孿生逐步更新；需要顯示完整
撤離時間軸時再使用 `/pareto`。

## 前端 request

完整可執行假資料位於
[`examples/digital_twin_request.json`](../examples/digital_twin_request.json)。

~~~json
{
  "request_id": "demo-typhoon-2026-001",
  "observed_at": "2026-08-01T08:00:00+08:00",
  "closure_at": "2026-08-01T12:00:00+08:00",
  "resources": {
    "tug_capacity": 8,
    "entrance_capacity": {"1": 1, "2": 1}
  },
  "vessels": [
    {
      "ship_id": "demo-01",
      "name": "DEMO VESSEL 01",
      "gross_tonnage": 8200,
      "entrance": "1",
      "ready_at": "2026-08-01T08:00:00+08:00",
      "transit_hours": 0.45,
      "tugs": 1,
      "risk_points": 2
    }
  ],
  "continuous_preferences": [
    {
      "key": "operations-balanced",
      "label": "營運均衡",
      "weights": [0.34, 0.33, 0.33]
    }
  ]
}
~~~

正式 request 的 `vessels` 目前必須剛好有 30 艘。欄位意義如下：

| 欄位 | 驗證與語意 |
|---|---|
| `request_id` | 每個 snapshot 的唯一 ID；回傳會原樣帶回 |
| `observed_at` | 此次決策 epoch，必須含 UTC offset |
| `closure_at` | 停止出港時間，必須比 observed time 晚 2–12 小時 |
| `resources.tug_capacity` | 2–12 艘可用拖船 |
| `resources.entrance_capacity` | 現有模型固定 `{"1": 1, "2": 1}` |
| `ship_id` | 船舶唯一 ID |
| `gross_tonnage` | 正數 GT |
| `entrance` | `"1"` 或 `"2"` |
| `ready_at` | 預估完成離港整備時間，含 UTC offset |
| `transit_hours` | 入口資源被占用的 service time；訓練近似範圍 0.36–1.65 小時 |
| `tugs` | 該船出港需求 1–4 艘拖船 |
| `risk_points` | 1–5；目前由呼叫端定義，不應宣稱為港方正式風險順位 |
| `risk_components` | 選填、只做追蹤的風險明細 |
| `weights` | `[count, GT, risk]`；非負且總和大於 0，後端會 normalization |

目前後端接受的是「已補齊」snapshot，不會自行猜測缺值。之後的
`/digital-twin/resolve` 才負責把缺失的 GT、service time、拖船與風險欄位補成 estimated，
並標示 observed／estimated。

## JavaScript 呼叫

~~~js
const response = await fetch(
  `${TYPHOON_API_BASE}/api/typhoon/digital-twin/step`,
  {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(snapshot),
  },
);

if (!response.ok) {
  throw new Error(`Typhoon RL ${response.status}: ${await response.text()}`);
}

const result = await response.json();
~~~

開發階段可先把 `examples/digital_twin_request.json` 載入成 `snapshot`。正式整合時，以空拍圖
或港區資料中的在港船舶取代假船，但 request shape 不變。

## `/step` response

~~~json
{
  "request_id": "demo-typhoon-2026-001",
  "engine": "python-rl-multi-pareto",
  "pareto_count": 3,
  "next_actions": [
    {
      "candidate_id": "soft-moe:risk",
      "preference": {
        "key": "risk",
        "weights": [0.15, 0.15, 0.70]
      },
      "next_decision": {
        "type": "dispatch",
        "time": 0.0,
        "vessel": {
          "ship_id": "demo-05",
          "start_hour": 0.0,
          "finish_hour": 0.95
        }
      },
      "projected_kpi": {
        "count": 12,
        "gt": 420000,
        "risk": 43,
        "safety_violations": 0,
        "rejected_actions": 0
      }
    }
  ]
}
~~~

以上數字只是 response shape 範例，不是固定推論結果。前端應依欄位處理：

- `type = dispatch`：將指定 `ship_id` 排入出港；動畫路徑由數位孿生依泊位與航道座標產生。
- `type = wait`：維持現況至 `until`，等待船舶整備或資源釋放。
- `time`、`start_hour`、`finish_hour`：相對於 `observed_at` 的小時數。
- `projected_kpi`：若採用該候選並完成整段排程時的預估結果。

每當船舶真的出港、整備延誤、拖船變動或封港時間改變，前端應送新的 snapshot 再呼叫
`/step`，而不是把第一次產生的完整排程永久鎖死。

## 目前限制

- 固定 30 艘船；多於 30 艘與少於 30 艘的正式 adapter 尚未納入服務。
- 目前只驗證 seed 42，屬 RL Pareto PoC。
- service time、拖船與風險點若用假資料，只能作展示。
- action mask 保證模型不主動違反封港截止、整備、入口與拖船容量，但不取代人員核准。
- CORS 預設 `*` 方便本機整合，對外部署前應設定可信任 origin。
