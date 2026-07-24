# 偏好學習實作流程與 commit 規劃

## 目標

讓 count、balanced、GT、risk 四種偏好在「確實存在取捨」的狀態下學到不同決策，同時
保持封港截止、入口容量與拖船容量為不可交換的硬限制。

## Commit 1：獨立專案骨架

範圍：`README.md`、`pyproject.toml`、`.gitignore`、`data/README.md`、既有颱風程式、前端、
測試與研究文件。

驗收：Git 不追蹤 SQLite、模型或報告；`pytest -q` 可在沒有大型資料檔時執行單元測試。

建議訊息：`chore: extract standalone typhoon evacuation PoC`

## Commit 2：Phase 0 多目標資料契約

範圍：

1. 每一步保存 `[count, GT, risk]` objective reward；
2. scalar PPO 只在最外層依 preference 加權；
3. 提供 vector-reward wrapper，供 MORL 演算法使用；
4. 風險 proxy 改成與 GT 解耦、可重現且明確標記 synthetic 的欄位；
5. 新增對應測試。

驗收：vector reward shape 固定為 `(3,)`；scalar reward 等於偏好與 vector reward 的內積；
安全違規維持 0；GT-risk correlation 不再由定義保證接近 1。

建議訊息：`feat: expose vector rewards and independent synthetic risk`

## Commit 3：配對偏好評估

範圍：同一 scenario、closure、tugs、compression、jitter、seed 下執行四種偏好；輸出
schedule unique count、cross-utility matrix 與每組偏好的 action trace。

驗收：同一 case 的四偏好 seed 必須相同；cross-utility matrix 可重現；跨偏好差異不能由
不同 jitter 解釋。

建議訊息：`feat: add paired preference evaluation diagnostics`

## Commit 4：四個 specialist

範圍：固定 count、balanced、GT、risk 的四模型訓練入口與 manifest。

用途：specialist 是環境可學性的診斷上限，不是最終部署架構。如果四個 specialist 仍然
重合，應先修正情境衝突，而不是增加 conditioned PPO steps。

驗收：四個模型輸出到固定且被忽略的子目錄；每個 manifest 記錄 profile、權重、seed、
steps、train/test dates；以至少三個 seed 執行正式比較。

評估命令：`.venv/bin/python -m typhoon.evaluate_specialists --db data/ua1008l.sqlite --seed 42`。

建議訊息：`feat: train fixed-preference specialist policies`

## Commit 5：Conditioned PPO 改造（Phase 1 已完成）

已加入四 profile 均衡 rollout、deterministic paired scenario stream、獨立 preference encoder
與 FiLM vessel-feature gating。Seed 42 結果見 `PHASE1_RESULTS_SEED42.md`：行為分離明顯改善，
但只有 count 達成 cross-utility 對角最佳，因此下一步不是直接擴多 seed，而是分解 objective
critic 與 policy loss。

## Commit 6：Decomposed PPO（Phase 2 已完成）

已加入 vector rollout、三頭 critic、逐目標 GAE、逐目標 PPO clipping、late scalarization 與
gradient cosine diagnostics。Seed 42 結果見 `PHASE2_RESULTS_SEED42.md`：GT credit assignment
明顯改善，但 cross-utility 仍只有 count 對角最佳。Post-hoc 分布確認 count–GT、GT–risk
分別有 38%、40% 負 cosine，因此進入 PCGrad 受控消融；Envelope Q-learning 繼續保留為
獨立 benchmark。

建議訊息：`feat: add decomposed multi-objective PPO`

## Commit 7：PCGrad 受控消融（Phase 3 已完成）

PCGrad 只作用於 shared actor/extractor 的 preference-weighted objective policy gradient；critic、
entropy、資料順序、訓練量與 hard action mask 維持不變。Seed 42 投影比例為 26.43%，但
cross-utility 對角最佳由 1/4 降為 0/4，GT utility 由 0.3633 降為 0.3486。結果證明負 cosine
包含有意義的 MORL 取捨，不能一律視為有害干擾。完整結果見 `PHASE3_RESULTS_SEED42.md`。

本 checkpoint 不取代 Phase 2。下一個研究步驟是 shared trunk + preference-specific heads／
soft mixture-of-experts 的 actor routing ablation。

建議訊息：`feat: add PCGrad policy-gradient ablation`

## Commit 8：Hard preference heads（Phase 4 已完成）

保留 shared preference-gated extractor、actor trunk 與三頭 critic，只將 action output 分成
四個相同初始化的 preference heads。行為差異提升至 59/60，52/60 cases 的四種 action trace
全不同；但 cross-utility 仍只有 count 對角最佳。count head 甚至在四種評分下全部最高，
顯示其他 heads 的 policy region 學習不足。完整結果見 `PHASE4_RESULTS_SEED42.md`。

四個獨立 specialist 本身只有 GT、risk 對角最佳，因此後續先以恢復這 2/4 與降低 specialist
regret 為門檻。下一個結構消融是 shared extractor + 四個完整 actor experts，不直接跳到
soft routing。

建議訊息：`feat: add hard preference-head actor routing`

## Commit 9：Full actor experts（Phase 5 已完成）

共享 preference-gated extractor 與三頭 critic，四個 profile 各自使用完整 256/128 actor
expert。50/60 cases 產生四種不同 action traces，但 cross-utility 對角最佳降為 0/4；GT expert
偏向 count、risk expert 偏向 GT、count expert 偏向 risk。映射稽核未發現 label wiring bug。
完整結果見 `PHASE5_RESULTS_SEED42.md`。

結果排除「共享 actor trunk 容量不足」作為主要原因。下一步不直接上 soft MoE，而是以同一套
custom decomposed PPO 分別訓練四個 fixed-profile models，隔離 multi-profile sharing 與 custom
optimizer/objective 的影響。

建議訊息：`feat: add full preference-expert ablation`

## Commit 10：Custom fixed-profile gate（Phase 6 已完成）

使用同一 custom decomposed PPO 分別訓練四個完全獨立的 fixed-profile models。四模型不共享
extractor、critic、optimizer 或 normalization，但 cross-utility 仍只有 balanced 對角最佳；
舊 SB3 specialists 已證明可達的 GT、risk 皆失敗。完整結果見 `PHASE6_RESULTS_SEED42.md`。

Soft MoE gate 因此未通過，暫不加入 router。問題已定位到 custom actor surrogate／update：
下一個最小修正是先依 preference scalarize vector advantage，再做一次 normalization 與 PPO
clipping；待 fixed-profile 恢復 GT、risk 2/4 後才重新考慮 soft MoE。

建議訊息：`feat: add pre-MoE fixed-profile diagnostic`

## Commit 11：Preference-first scalar surrogate（Phase 7 已完成）

保留 vector critic、vector returns 與 vector GAE，但 actor 先依 transition preference 合成
scalar advantage，再只做一次 normalization 與 PPO clipping。四個 fixed-profile models
在 seed 42 的對角最佳由 1/4 提升為 2/4，count、GT 成功對角；risk 仍由 count model 最高，
因此 soft MoE gate 尚未完全通過。完整結果見 `PHASE7_RESULTS_SEED42.md`。

下一個最小實驗應保留 Phase 7 actor surrogate，改測 preference-scalarized critic loss 加小權重
vector auxiliary loss。先只重跑 GT、risk；risk 恢復且 GT 不退步後，才重跑四偏好並考慮
soft MoE。

建議訊息：`feat: scalarize advantages before PPO clipping`

## Commit 12：Preference-scalarized critic（Phase 8 已完成）

在 Phase 7 actor surrogate 上，將 critic 主 loss 改成 preference-scalarized value MSE，並保留
0.1 倍 vector MSE auxiliary loss。四個 fixed-profile models 的 utility 全部提升；GT、risk
恢復各自 cross-utility 對角最佳，達成舊 SB3 specialists 的 2/2 gate。完整結果見
`PHASE8_RESULTS_SEED42.md`。

Expert gate 已通過，下一步可加入 soft MoE：以 Phase 8 experts 初始化，先凍結 experts 訓練
router，再以小 learning rate joint fine-tune；評估需加入 centroid 間的連續偏好。

建議訊息：`feat: align vector critic with preference utility`

## 整體 MORL 訓練完成後：30 艘容量與部署待辦

目前實驗繼續固定 30 艘，以免在 MORL 演算法比較期間同時改變 observation、action space 與
資料分布。部署前再處理以下事項：

1. 先以真實紀錄驗證「同一決策時點的在港待撤船數」分布；每日出港紀錄數不等於同時在港
   船數。若大部分時點接近或低於 30，保留 30-slot 容量即可，不必立刻改成任意長度模型。
2. 多於 30 艘時採 rolling top-30 candidate window；以 readiness、可行性與封港緊迫度等
   確定性規則選入，派船或事件推進後重新建立候選集。排序規則必須固定，避免 slot 漂移。
3. 少於 30 艘時補空 slot。依目前絕對時間定義，空位可暫設
   `ready_hour = now + 4 × (closure_hour - now)`，而不是只填相對剩餘時間。
4. 準備時間只能當額外防線；空 slot 必須有 `valid_vessel = 0`，對應 action 永久 mask，且
   count、GT、risk 的分母使用真實船舶，不把 padding 算進 KPI。
5. 這項變更需要以不同實際船數重新訓練與評估，不能把未看過 padding 的現有 checkpoint
   直接宣稱為可變船數模型。
