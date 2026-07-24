# 颱風撤離 RL：目前結論與比較口徑

## 一、目前能下的結論

本 PoC 已建立颱風封港前撤離環境、硬式安全遮罩、Phase 8 四個 fixed-profile experts、
Phase 9B 單一 Soft MoE checkpoint、規則基線與離線 Pareto 前端。API 預設依四種離散偏好
選擇 Phase 8 expert；實驗模式可送入任意三維非負偏好，由同一個 Phase 9B checkpoint
完成推論。模型能在未參與訓練的歷史日期上完成排程，所有受測策略的安全違規皆為 0。

目前 **不能** 宣稱 RL 全面優於規則方法。100,000-step 模型整體效用略高於 FCFS，
但仍低於直接依同一效用設計的 value-density；在風險偏好下，Risk-aware 的撤離 GT、
風險點與風險加權效用亦略高於 RL-risk。

這不是異常，而是多目標排程的正常現象：艘數、GT、風險點會互相競爭，專門針對單一
目標的規則可能位於 Pareto 前緣，RL 不可能被合理要求在所有原始指標上同時嚴格勝出。

## 二、風險加權效用是什麼

它不是新的氣象或航安量測，而是方便訓練和配對比較的合成分數。令：

- \(C\)：撤離艘數／情境總艘數；
- \(G\)：撤離 GT／情境總 GT；
- \(R\)：撤離風險點／情境總風險點。

風險偏好的效用為：

\[
U_{risk}=0.15C+0.15G+0.70R
\]

效用約介於 0 到 1，越高表示越符合該組偏好。安全違規不納入加權；封港截止、拖船
容量與入口衝突均由動作遮罩強制為硬限制。

舊版 100,000-step 結果使用依 GT 分級的 risk proxy；新獨立專案已改為與 GT 解耦、
可重現的 synthetic risk components。兩者不是港方正式危險度，且新舊報表不可直接混比。

## 三、正確的比較方式

所有方法必須使用相同船舶、封港時間、拖船容量、需求壓力、資訊範圍與安全限制。
不同目標應分開配對：

| 決策目標 | 主要比較 |
|---|---|
| 艘數優先 | RL-count vs throughput／value-density-count |
| GT 優先 | RL-GT vs GT-aware |
| 風險優先 | RL-risk vs Risk-aware／value-density-risk |
| 平衡偏好 | RL-balanced vs FCFS 與所有基線 |
| 多目標能力 | RL 非支配集合 vs 規則非支配集合 |

把四種偏好的結果混成單一平均，只能看整體泛化，不能用來宣稱某一專門目標勝負。

## 四、早期 100,000-step conditioned 模型結果

以下保留為 Phase 8 前的歷史基線。訓練使用 24 個歷史日期；最後 5 個日期完全保留。評估涵蓋 4 種偏好、3 個封港時間、
2 種拖船容量與 2 種需求壓力，共 240 回合／策略，且每一偏好有 60 回合。

### 風險偏好配對

| 方法 | 風險加權效用 | 95% CI 半寬 | 撤離艘數 | 撤離 GT | 撤離風險點 |
|---|---:|---:|---:|---:|---:|
| RL-risk | 0.3865 | 0.0395 | 12.35 | 310,972 | 18.78 |
| Risk-aware | 0.3911 | 0.0402 | 11.00 | 349,030 | 19.15 |
| Value-density-risk | 0.4098 | 0.0419 | 12.27 | 359,637 | 19.93 |

目前信賴區間高度重疊，尚無足夠證據宣稱方法間存在統計顯著差異。RL-risk 較偏撤離
更多艘船，Risk-aware 與 value-density-risk 則撤出較高 GT 與風險點。

### 各偏好的 RL 效用

| 偏好 | RL | FCFS | Risk-aware | Value-density |
|---|---:|---:|---:|---:|
| 艘數 | 0.3983 | 0.3964 | 0.3747 | 0.4316 |
| 平衡 | 0.3847 | 0.3781 | 0.3861 | 0.4098 |
| GT | 0.3644 | 0.3540 | 0.3858 | 0.4160 |
| 風險 | 0.3865 | 0.3860 | 0.3911 | 0.4098 |

模型對偏好已有輸入條件，但目前四種輸出行為的分離仍不明顯。100,000 steps 是第一個
正式 checkpoint，不是收斂證明；後續應加入多訓練 seed、學習曲線、偏好分離檢查與
最佳化上限，才能判斷增加訓練量是否真的有效。

## 五、Phase 8 expert gate

Phase 8 保留 preference-first scalar actor surrogate，並以 preference-scalarized critic MSE
加 0.1 倍 vector auxiliary MSE 訓練。Seed 42 held-out 結果如下：

| 偏好 | Phase 8 utility | 舊 SB3 specialist | Delta |
|---|---:|---:|---:|
| 艘數 | 0.4018 | 0.4031 | -0.0013 |
| 平衡 | 0.3813 | 0.3895 | -0.0082 |
| GT | 0.3669 | 0.3683 | -0.0014 |
| 風險 | 0.4005 | 0.4048 | -0.0043 |

GT、risk 均達成 cross-utility 對角最佳，通過 soft MoE 前的 2/2 expert gate。count、balanced
未對角不構成失敗，因為舊 SB3 specialist 亦如此，且 risk expert 在目前資料同時具有較高
count 與 risk completion。

## 六、前端與模型的界線

前端不自行模擬或偽造 RL。一次情境重算會呼叫 Python API，載入 Phase 8 四個
`final.pt`，依離散偏好 hard-route 至對應 expert，並以相同情境執行 FCFS、Risk-aware 與
value-density。若任一模型缺少或介面不相容，API 回傳 503，不會把 heuristic 靜默標示為
RL。

前端負責顯示完整 dispatch／wait 動作軌跡、時程、基線與 Pareto 結果。若多個 RL 偏好
產生完全相同的動作與 KPI，介面會明確標示結果重疊，而不製造不存在的 Pareto 差異。

Phase 9B 另提供預設關閉的連續偏好控制。它是四個 frozen experts、PPO-trained soft router
與向量 critic 的單一 checkpoint，並非已完成 joint fine-tuning 的單一共享策略。GT/risk gate
維持 2/2；15 點 grid 平均 utility delta 為 +0.001626，安全為 0，但 count／balanced 仍非
對角最佳且目前只有單 seed，因此只能標示為實驗結果，不能取代 Phase 8。

## 七、後續工作

- 以多個訓練 seed 確認偏好反應與結果穩定性；
- 加入 GT-aware 基線及最佳化上限；
- 將合成風險 proxy 替換為危險品、吃水、主機狀態等正式欄位；
- 重跑 Phase 9B seeds 43、44，並以 0.1 preference grid 加入 bootstrap CI 與 regret；
- 多 seed 通過後才延長 steps 或以小 learning rate joint fine-tune；
- 讓單一 Soft MoE 對同一情境輸出多組偏好解，再比較 RL 與規則方法的 Pareto 集合。

目前程式盤點、偏好未分離的原因、MORL 技術選型與分階段驗收方式，見
[typhoon_preference_learning_plan.md](typhoon_preference_learning_plan.md)。

上述工作在宣稱颱風 RL 優於規則前必須完成。
