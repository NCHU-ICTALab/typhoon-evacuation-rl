# Typhoon Evacuation RL

颱風封港前船舶撤離排序的離線決策輔助 PoC。系統在預估停止出港前，依船舶整備狀態、
港口入口與拖船容量，產生可執行的派船／等待順序，並讓操作人員調整「撤離艘數、總噸位
（GT）、風險點」三項目標的偏好。

> 本系統不發布封港命令，也不取代港長、VTS、引水人或船長。現有氣象、資源與風險欄位
> 含合成研究假設，只能用於演算法與流程展示。

本專案的服務目標是由 **RL 產生多個偏好特化的 Pareto 排程候選**，讓操作人員或數位孿生
選擇，而不是把 Value-density 規則直接當成正式主策略。Value-density、FCFS 與 Risk-aware
保留為離線評估基準；數位孿生的 RL 端點不會把規則候選混入 RL Pareto 集。

## 應用情境

一般交通量下，固定規則通常已能完成大部分排程；RL 的價值主要出現在封港倒數、資源不足、
船舶集中整備且目標互相衝突時。例如：多派小船可能提高撤離艘數，優先大型船提高撤離 GT，
優先危險品或機動受限船則提高風險點覆蓋。

模型每一步只能選擇：

- `A0–A29`：派指定船舶出港；
- `A30`：等待下一艘船整備完成或資源釋放。

封港截止、入口同時容量、拖船容量、船舶整備與過航完成時間均由 action mask 強制限制，
不能以較高 reward 換取違規。

## RL 與 Router 架構

目前的連續偏好模型是一個 Soft Mixture-of-Experts checkpoint：四個 frozen experts 分別代表
艘數、平衡、GT 與風險取向；soft router 依當下狀態與操作人員輸入的三維偏好，決定四個
experts 的混合比例。

~~~text
偏好權重 (count, GT, risk) ──► preference router ─┐
                                                    ├─► softmax ─► 4 個 expert 權重
全域時間／資源狀態 ───────────► state router ──────┘

30 艘船舶狀態 ─► 4 個 frozen experts ─► masked action probabilities
                                             │
4 個 expert 權重 ────────────────────────────┴─► 機率混合 ─► 派船／等待

船舶、資源與偏好狀態 ─► vector critic ─► count／GT／risk 三維 value
~~~

訓練時保留三維 reward、return 與 GAE，再依每筆 transition 的偏好合成 scalar advantage，做一次
normalization 與 PPO clipping。Router 與 vector critic 會更新，四個 experts 維持凍結；安全
action mask 會在 expert 內與混合後各套用一次。

## 訓練資料集

專案採完全離線資料流程，不需要持續抓取或即時 API。

| 資料類別 | 目前使用內容 | 用途與界線 |
|---|---|---|
| 靜態船舶歷史快照 | `ua1008l.sqlite` 中的出港日期、引水申請時間、船名、GT、碼頭與港口入口 | 唯讀建立 30 艘船情境；資料庫不進 Git，也不代表颱風當下實際滯港船舶 |
| 公開靜態規則 | 颱風期間達出港管制風力門檻後停止出港 | 編成不可違反的封港截止條件 |
| 衍生欄位 | 由歷史申請時間換算整備時間；由 GT 查表產生過航時間與拖船需求 | 研究用假設，不是實測港口容量 |
| 合成情境 | 封港倒數、拖船容量、需求壓力、時間 jitter | 擴充訓練分布與高壓封港案例 |
| 合成風險 proxy | 危險品、吃水、機動性、主機整備等可重現 proxy | 與 GT 解耦，但不是港方正式風險或優先序 |

目前使用 24 個較早日期訓練，最後 5 個日期完全保留作 held-out 評估。每個 episode 會抽樣歷史
日期、封港時間、拖船容量、需求壓力與 jitter；偏好資料由 50% 固定目標 centroid 與 50%
`Dirichlet(0.7)` 連續權重組成。不同方法比較時使用相同船舶、資源、隨機 seed 與安全限制。

資料庫預設位置為 `data/ua1008l.sqlite`，程式以 SQLite read-only mode 開啟，不會複製或修改。
資料的散布、授權與敏感性仍應在公開 GitHub 前獨立確認；詳見 [data/README.md](data/README.md)。

## 目前結果

### RL vs FCFS

在最後 5 個未參與訓練的日期與 60 組完全配對封港情境中，RL 與 FCFS 使用相同船舶、
封港時間、拖船容量、需求壓力、jitter seed 與 hard action mask。四種主要偏好的 held-out
weighted utility 如下：

| 偏好 | RL weighted utility | FCFS | 相對提升 |
|---|---:|---:|---:|
| 艘數 | 0.4046 | 0.3990 | +1.40% |
| 平衡 | 0.3915 | 0.3814 | +2.65% |
| GT | 0.3650 | 0.3575 | +2.10% |
| 風險 | 0.4053 | 0.3878 | +4.51% |

為評估連續偏好，另在 count／GT／risk simplex 掃描 15 個間距 `0.25` 的權重，共完成
900 組 RL／FCFS 配對：

- 15/15 偏好點的平均 utility 均高於 FCFS；
- 跨偏好平均絕對提升 `+0.009395`，相對提升 `+2.46%`；
- 逐 episode 勝／平／負為 403／227／270；
- 同一情境平均產生 5.38 種 RL 排程，FCFS 固定為 1 種；
- 每個情境平均保留 3.23 個非支配 RL 候選；
- 25/60 cases 至少有一個 RL 候選 Pareto 支配 FCFS；
- 相鄰 `0.25` 偏好點有 46.94% 會改變排程；safety violations 與 rejected actions 均為 0。

「連續偏好」表示 API 可接受任意非負三維權重；排程動作仍是離散的，因此不保證船序隨權重
平滑插值。完整評估方法、原始 KPI、限制與報告措辭見
[Phase 14 FCFS comparison](docs/PHASE14_FCFS_COMPARISON.md)。

### 研究與服務界線

Phase 11 的最佳化 oracle 顯示 Value-density 在目前近可分離的合成環境已距上限約 1–3%。
這項結果用來界定研究天花板，**不代表服務策略改成 Value-density**；數位孿生端點只回傳
RL 產生的多偏好 Pareto 候選。

目前量化結果只有 seed 42，且真實 service time、可變船數、多 seed 與 paired bootstrap CI
尚未完成，因此應表述為 held-out RL Pareto PoC，不是正式港務生產驗收。

Phase 13 已提供固定 30 艘、時區明確的 stateless snapshot 契約，涵蓋 `observed_at`／
`closure_at`、船舶整備與 service time、拖船與入口容量。呼叫端可取得完整 RL Pareto 排程，
或每個 Pareto 候選的下一步動作；VD 不在此端點的候選集合中。

## 快速開始

~~~bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q

# 從 VD-DAgger warm-start 訓練 preference-conditioned PPO
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.train_vd_warmstart_ppo \
  --db data/ua1008l.sqlite --seed 42 --expert-steps 25000 \
  --router-steps-per-profile 25000

# 連續偏好配對評估
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.evaluate_soft_moe_grid \
  --db data/ua1008l.sqlite --step 0.25 --seed 42

# 啟動 Python API 與前端
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TYPHOON_DB_PATH=/path/to/ua1008l.sqlite \
  .venv/bin/python -m typhoon.api
~~~

啟動後開啟 `http://127.0.0.1:8765`。前端預設使用四個離散 expert 的穩定路徑；開啟
「實驗性連續偏好」後，才會使用 PPO-trained Soft MoE router。

數位孿生能力宣告位於 `GET /api/typhoon/digital-twin/capabilities`；RL-only 完整 Pareto
排程與 receding-horizon 下一步建議分別使用 `POST /api/typhoon/digital-twin/pareto` 與
`POST /api/typhoon/digital-twin/step`。契約與目前不支援的狀態見
[docs/PHASE13_DIGITAL_TWIN.md](docs/PHASE13_DIGITAL_TWIN.md)。

## 專案內容

~~~text
typhoon/       環境、RL、數位孿生契約、規則基線、API 與前端
tests/         action mask、PPO、router、API 與資料契約測試
docs/          實驗規約、結果與研究歷程
data/          本機唯讀資料說明；資料本身不進 Git
models/        本機 checkpoint 與評估輸出；不進 Git
~~~

目前比較口徑與限制見
[docs/typhoon_rl_current_conclusions.md](docs/typhoon_rl_current_conclusions.md)，訓練與評估規約見
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)，完整研究歷程則保留在
[docs/WORKFLOW.md](docs/WORKFLOW.md)。
