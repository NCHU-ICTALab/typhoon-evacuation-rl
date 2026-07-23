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
明顯改善，但 cross-utility 仍只有 count 對角最佳。下一個最小實驗是 gradient conflict 分布
與 shared actor 對照 preference heads；Envelope Q-learning 繼續保留為獨立 benchmark。

建議訊息：`feat: add decomposed multi-objective PPO`
