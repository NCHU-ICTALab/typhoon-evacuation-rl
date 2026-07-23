# 颱風撤離：程式盤點與偏好學習計畫

## 1. 目前程式分層

目前 PoC 有兩條共用同一批靜態船舶模板的執行路徑。

### 規則與報表路徑

~~~text
build_scenario.py
  └─ 讀取 ua1008l.sqlite，建立船舶、合成風力與資源情境
rules.py
  └─ 計算封港硬截止
planner.py
  └─ FCFS、risk-aware 規則排程
run_poc.py
  └─ 產生 out/latest_scenario.json、latest_report.json、latest_report.md
~~~

### RL、比較與前端路徑

~~~text
build_scenario.py
  └─ 多日期情境池
env.py
  └─ 30 艘船、31 個離散動作、硬式 action mask
train_rl.py
  └─ MaskablePPO 訓練與固定檔名模型輸出
evaluate_rl.py + baselines.py
  └─ RL、FCFS、risk-aware、value-density 離線比較
api.py
  └─ 同一請求執行四種 RL 偏好及三種基線
frontend/
  └─ 顯示時程、完整動作軌跡、Pareto 與重疊結果
~~~

測試分成 `test_poc.py`（規則與 planner）、`test_env.py`（環境與遮罩）及
`test_api.py`（Python RL API 與禁止靜默 fallback）。`models/` 與 `out/` 都使用固定
名稱覆寫且不進 Git，不會形成持續累積的抓取工作負載。

## 2. 現行偏好 RL 的實際做法

目前使用一個共享的 MaskablePPO：

- 動作 `A0` 至 `A29` 是現在派指定船舶出港，`A30` 是等待下一個事件；
- observation 是每艘船 11 個特徵乘 30 艘，加 14 個全域特徵，共 344 維；
- 全域特徵最後三維是艘數、GT、風險的偏好權重；
- 訓練時以 `Dirichlet(1,1,1)` 隨機抽偏好；
- 每一步先產生 `[count, GT, risk]` vector reward，再依偏好取內積供目前 PPO 使用：

\[
r=10\left(
w_C\frac{1}{N}
+w_G\frac{GT_i}{GT_{total}}
+w_R\frac{Risk_i}{Risk_{total}}
\right)+\text{wait cost}
\]

安全不參與加權；不符合封港截止、拖船或入口容量的動作直接由 action mask 移除。

## 3. 為何四種偏好容易產生相同排程

原始版本不應只解讀成「訓練量不足」，診斷出四個更基本的原因：

1. **過早標量化**：舊版三目標在進入 PPO buffer 前只留下 scalar reward，單一 value
   head 無法分別保存每個目標的 credit assignment。新專案已在 `info` 與 wrapper 保留向量。
2. **偏好訊號太弱**：偏好只佔 344 維 observation 的三維，網路尚無獨立 preference
   encoder、gating 或多頭 value。
3. **目標原本不夠衝突**：舊版風險由 GT 分級，29 日、870 艘的相關係數為 **0.9343**。
   Phase 0 改成獨立 deterministic synthetic profile 後降為 **0.0396**。
4. **可選動作不足**：部分時間點只有一艘同入口船可行；這種狀態採相同動作才是正確。

舊版跨偏好評估也使用不同 seed。新專案已改成同一 case 的日期、資源、jitter 與 seed
完全相同，並輸出 action trace unique count 與 cross-utility matrix。

## 4. 可採用的相關技術

- [Dynamic Weights in Multi-Objective Deep Reinforcement Learning
  (ICML 2019)](https://proceedings.mlr.press/v97/abels19a.html)：以偏好權重 conditioning
  多目標 Q-network，並用 Diverse Experience Replay 維持不同權重經驗。概念適合本案，
  但原方法是 off-policy Q-learning，不是現行 PPO 的直接替換。
- [Envelope Q-learning
  (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/4a46fbfca3f1465a27b210f4bdfe6ab3-Abstract.html)：
  保留 vector Q-value，依偏好在 convex coverage set 上選解。本案只有 31 個離散動作，
  型態相符；但需另整合 action mask。
- [MORL-Baselines / MO-Gymnasium
  (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4aa8891583f07ae200ba07843954caeb-Abstract-Datasets_and_Benchmarks.html)：
  提供 vector reward API、Envelope Q-learning、Pareto Conditioned Networks 與 Pareto
  評估工具，適合定義本案介面和對照實驗。
- [D3PO (2026 preprint)](https://arxiv.org/abs/2602.07764)：針對偏好條件 PPO 的過早
  scalarization 與 representation collapse，採多目標 value、逐目標 advantage／loss、
  延後偏好加權及 diversity regularizer。它最接近現有路線，但仍是新預印本，應視為
  自行實作的實驗方案，而非成熟套件。

## 5. 建議實作順序

### Phase 0：資料契約與可學性診斷（第一批已完成）

- [x] 保存 `[count, GT, risk]` vector reward，舊 PPO 再依 preference scalarize。
- [x] 以相同日期、資源、jitter 與 seed 評估四種偏好。
- [x] 提供 count、balanced、GT、risk 四個固定權重 specialist 訓練入口。
- [x] 將 risk proxy 與 GT 解耦並標明 synthetic。
- [ ] 擴充更多同時有多個可行、且目標互相衝突的壓力案例。

如果四個 specialist 仍產生相同排程，問題在情境或目標；增加 conditioned policy 的
訓練步數不會解決。

### Phase 1：保留 PPO 的低風險改造

1. 每個 rollout 固定平均包含四種營運偏好，不只依賴 Dirichlet 抽樣。
2. 為偏好建立獨立 encoder，透過 feature gating／FiLM 影響船舶特徵。
3. 若產品只需要四種固定偏好，可先採共享 trunk 加四個 policy/value heads。
4. 以至少 3 至 5 個 seed 比較 specialist 與 conditioned PPO，再決定是否加大 steps。

### Phase 2：真正的多目標演算法

- 延續 PPO：做 D3PO-lite，加入 vector rollout、三頭 critic、逐目標 GAE／clipped loss，
  以及只在有多個可行衝突動作時施加 masked-logit diversity regularizer；或
- 採成熟的離散 MORL 對照：使用 Envelope Q-learning，為 31 個動作加入安全遮罩。

不建議只把現行 scalar PPO 從 100,000 提升到 1,000,000 steps；那可能只是更穩定地收斂
到同一個折衷策略。

## 6. 驗收與比較口徑

- 安全違規與 rejected unsafe action 必須為 0；
- 同情境、同 seed 的四偏好 schedule/action-trace 唯一數；
- 同一 observation 下，各偏好 masked action distribution 的 JS divergence；
- cross-utility matrix：用每組權重評分所有偏好排程，檢查對角線是否最佳或近最佳；
- conditioned policy 相對於相同偏好 specialist 的 utility regret；
- Pareto hypervolume、epsilon indicator、非支配解數與解間稀疏度；
- 多訓練 seed 的平均、信賴區間及 paired bootstrap。

RL 不需要在每個原始指標上同時擊敗所有專門規則。合理目標是：在相同安全限制下，對
指定偏好的效用接近或勝過對應 specialist／規則，並以一個模型覆蓋多種偏好與未見情境。
