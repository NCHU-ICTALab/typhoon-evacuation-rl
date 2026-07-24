# Phase 8 推理整合 v1

## 目前架構

前端不再載入舊 `final.zip`。FastAPI 啟動時會載入 Phase 8 的 count、balanced、GT、risk
四個 `final.pt`，每次請求在相同情境上各執行一次：

~~~text
discrete preference
→ fixed-profile hard selector
→ Phase 8 expert
→ action-masked deterministic schedule
~~~

API 回傳 `expert_profile`、完整 dispatch/wait trace、KPI、Pareto candidates，以及 Phase 8
held-out validation record。任何 checkpoint 缺少或 observation/action space 不一致都會回傳
503，不會改用規則方法冒充 RL。

正式 comparison report 保持 Git ignored；精簡且可追蹤的驗證紀錄保存於
`typhoon/model_cards/phase8_seed42.json`，讓前端、文件與程式版本使用同一組數字。

## 實際推理 smoke record

2026-07-08 離線保留日、封港倒數 4 小時、8 艘拖船、需求壓力 3×：

| Expert | 撤離艘數 | 撤離 GT | 風險點 | 情境效用 | Safety |
|---|---:|---:|---:|---:|---:|
| count | 9 | 180,593 | 17 | 0.277296 | 0 |
| balanced | 9 | 219,391 | 16 | 0.259833 | 0 |
| GT | 9 | 201,294 | 17 | 0.244144 | 0 |
| risk | 10 | 176,858 | 20 | 0.277915 | 0 |

此單一情境用於端到端 smoke test，不取代 5 個 held-out dates、每 profile 60 cases 的正式
Phase 8 paired evaluation。正式紀錄仍是 GT/risk 2/2 對角最佳、safety violations 0。

## 啟動

~~~bash
cd /home/ubuntu/k7/typhoon-evacuation-rl
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TYPHOON_DB_PATH=/home/ubuntu/k7/sdci_data/ua1008l.sqlite \
/home/ubuntu/k7/.venv/bin/python -m typhoon.api
~~~

開啟 `http://127.0.0.1:8765`。Health endpoint 為 `/api/typhoon/health`。

## Soft MoE 相容介面

v1 暫時保留離散 `preference=count|balanced|gt|risk`。下一版加入 soft MoE 時：

1. request 新增可選的 `preference_weights: [count, gt, risk]`；
2. response 將 `model.routing` 改為 `soft-moe`；
3. 每個 RL result 新增 `routing_weights`，保留既有 schedule、decisions 與 KPI；
4. 四個 centroid request 仍須向下相容；
5. 前端沿用目前比較卡與時程，只增加連續偏好控制與 expert mixture 顯示。

這使 soft MoE 可以替換後端模型，而不必重寫排程畫面。
