# Typhoon Evacuation RL

颱風封港前船舶撤離排序的離線決策輔助 PoC。系統在預估停止出港前，依船舶整備狀態、
港口入口與拖船容量，產生可執行的派船／等待順序，並讓操作人員調整「撤離艘數、總噸位
（GT）、風險點」三項目標的偏好。

> 本系統不發布封港命令，也不取代港長、VTS、引水人或船長。現有氣象、資源與風險欄位
> 含合成研究假設，只能用於演算法與流程展示。

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

目前的 seed-42 router PPO pilot 使用 100,000 個 RL transitions。在 15 個連續偏好 grid 點、
每點 60 組 held-out cases 上，相較未經 RL 更新的 router 為 4 點改善、9 點持平、2 點小幅退步；
平均 utility delta 為 `+0.001626`，安全違規與 rejected actions 都是 0。

這只證明單一 seed 的連續偏好 router 已能改變排程並保有安全 gate，不代表 RL 已全面勝過
FCFS、Risk-aware 或 Value-density，也不是正式港務成效。

## 快速開始

~~~bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q

# 訓練目前的 frozen-expert continuous router
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.train_soft_moe_ppo \
  --db data/ua1008l.sqlite --seed 42 --steps-per-profile 25000

# 連續偏好配對評估
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.evaluate_soft_moe_grid \
  --db data/ua1008l.sqlite --step 0.25 --seed 42

# 啟動 Python API 與前端
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TYPHOON_DB_PATH=/path/to/ua1008l.sqlite \
  .venv/bin/python -m typhoon.api
~~~

啟動後開啟 `http://127.0.0.1:8765`。前端預設使用四個離散 expert 的穩定路徑；開啟
「實驗性連續偏好」後，才會使用 PPO-trained Soft MoE router。

## 專案內容

~~~text
typhoon/       環境、RL、規則基線、API 與前端
tests/         action mask、PPO、router、API 與資料契約測試
docs/          實驗規約、結果與研究歷程
data/          本機唯讀資料說明；資料本身不進 Git
models/        本機 checkpoint 與評估輸出；不進 Git
~~~

目前比較口徑與限制見
[docs/typhoon_rl_current_conclusions.md](docs/typhoon_rl_current_conclusions.md)，訓練與評估規約見
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)，完整研究歷程則保留在
[docs/WORKFLOW.md](docs/WORKFLOW.md)。
