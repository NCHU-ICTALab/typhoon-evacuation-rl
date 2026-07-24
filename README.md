# Typhoon Evacuation RL

封港前船舶撤離排序的離線決策輔助 PoC。專案只保留颱風情境、規則基線、RL、API、前端、
測試與研究文件；不包含原始資料庫、模型 checkpoint 或執行輸出。

> 本系統不發布封港命令，也不取代港長、VTS、引水人或船長。風力、資源與風險欄位含
> 合成研究假設，不能直接解讀為真實港務成效。

## 專案結構

~~~text
typhoon/                 Python package、API 與前端
tests/                    不依賴大型資料庫的單元測試
docs/                     資料界線、現況結論與偏好學習流程
data/README.md            外部 SQLite 放置方式；資料本身不進 Git
models/                   執行時產生；不進 Git
out/                      執行時產生；不進 Git
~~~

## 安裝與測試

~~~bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
~~~

## 準備離線資料

將 `ua1008l.sqlite` 放在 `data/ua1008l.sqlite`，或在命令列使用 `--db /path/to/file.sqlite`。
資料庫只以 SQLite read-only mode 開啟，不會被複製或修改。

## 執行

~~~bash
# 規則 PoC
.venv/bin/python -m typhoon.run_poc --db data/ua1008l.sqlite

# 原有 conditioned PPO
.venv/bin/python -m typhoon.train_rl --db data/ua1008l.sqlite --profile conditioned --steps 100000

# Phase 0：四個固定偏好 specialist
.venv/bin/python -m typhoon.train_specialists --db data/ua1008l.sqlite --steps 100000

# 評估 conditioned 模型
.venv/bin/python -m typhoon.evaluate_rl --db data/ua1008l.sqlite

# 評估四個 specialist（相同 held-out cases）
.venv/bin/python -m typhoon.evaluate_specialists --db data/ua1008l.sqlite --seed 42

# Phase 1：四偏好均衡 rollout + preference gating
.venv/bin/python -m typhoon.train_conditioned_v2 --db data/ua1008l.sqlite --steps-per-profile 100000 --seed 42

# Phase 2：decomposed critic + per-objective GAE/PPO
.venv/bin/python -m typhoon.train_decomposed_ppo --db data/ua1008l.sqlite --steps-per-profile 100000 --seed 42

# Phase 3：PCGrad 受控消融（結果不取代 Phase 2）
.venv/bin/python -m typhoon.train_decomposed_ppo --db data/ua1008l.sqlite --steps-per-profile 100000 --seed 42 --gradient-surgery pcgrad

# Phase 4：共享 trunk + 四個 hard preference heads
.venv/bin/python -m typhoon.train_decomposed_ppo --db data/ua1008l.sqlite --steps-per-profile 100000 --seed 42 --actor-routing hard_heads

# Phase 5：共享 extractor + 四個完整 actor experts
.venv/bin/python -m typhoon.train_decomposed_ppo --db data/ua1008l.sqlite --steps-per-profile 100000 --seed 42 --actor-routing full_experts

# Phase 6：soft MoE 前的 custom fixed-profile gate
for profile in count balanced gt risk; do
  .venv/bin/python -m typhoon.train_decomposed_ppo --db data/ua1008l.sqlite --steps-per-profile 100000 --seed 42 --batch-size 64 --fixed-profile "$profile" --skip-evaluation
done
.venv/bin/python -m typhoon.evaluate_decomposed_specialists --db data/ua1008l.sqlite --seed 42

# 前端與 Python API
.venv/bin/python -m typhoon.api
~~~

完整執行順序與 commit 建議見 [docs/WORKFLOW.md](docs/WORKFLOW.md)。PCGrad 技術契約與
seed 42 結果分別見 [docs/PHASE3_PCGRAD.md](docs/PHASE3_PCGRAD.md) 與
[docs/PHASE3_RESULTS_SEED42.md](docs/PHASE3_RESULTS_SEED42.md)。Hard-head routing 的技術與
結果見 [docs/PHASE4_HARD_HEADS.md](docs/PHASE4_HARD_HEADS.md) 與
[docs/PHASE4_RESULTS_SEED42.md](docs/PHASE4_RESULTS_SEED42.md)。Full actor experts 的技術與
結果見 [docs/PHASE5_FULL_EXPERTS.md](docs/PHASE5_FULL_EXPERTS.md) 與
[docs/PHASE5_RESULTS_SEED42.md](docs/PHASE5_RESULTS_SEED42.md)。Soft MoE 前置 gate 見
[docs/PHASE6_FIXED_PROFILE_DIAGNOSTIC.md](docs/PHASE6_FIXED_PROFILE_DIAGNOSTIC.md) 與
[docs/PHASE6_RESULTS_SEED42.md](docs/PHASE6_RESULTS_SEED42.md)。
