# Phase 11：競爭力定案（RL vs 規則 vs 嚴格最佳化上限）

## 目的

回答 Goal 2 的定案問題：在保有連續偏好的前提下，RL 能否**接近或打平**最強的規則方法。
把口徑從「勝過」改為「接近／打平」，並用一個**嚴格最佳化上限（oracle）**量出天花板，判斷
RL 與 value-density 的差距是否有意義、是否可收斂。

## 嚴格 oracle 的建構

關鍵性質：weighted utility 只取決於「哪些船被撤離」這個**集合**——每艘船貢獻固定非負值
`v_i = pref · [1/N, gt_i/總GT, risk_i/總risk]`。因此最佳解 = **最大權重的可排程子集**，受與
env 完全相同的硬限制（整備時間、封港截止、拖船與入口容量隨時間）。

`typhoon/evaluate_oracle.py` 以 branch-and-bound 求解：

- 以純陣列鏡射 env 動態（不 deepcopy env），大幅加速；
- 以 value-density 軌跡當初始下界（強 incumbent）；
- admissible 上界 = 目前效用 + 所有「時間上仍可趕上封港」的未排船 `v_i`（放寬拖船／入口
  contention），用於剪枝；
- **狀態去重**：dispatch 不推進時間，故「先派 A 再派 B」與「先 B 再 A」到達同一狀態，
  `(已撤集合, now, 進行中作業)` 只展開一次，collapse 掉所有排列；
- node cap 到達即停並標記 `proven_optimal=false`。

**下界性質**：search 回傳的每個值都是一個可實現的排程，因此永遠是真實最佳解的**有效下界**。
只要 oracle > VD，就已「證明」該 case 在 VD 之上存在 headroom（即使全域最佳未證）；封港緊迫
的 case 通常能直接證明全域最佳。單元測試以 brute-force 小案例交叉驗證（撤離兩艘最大船
= 50k/60k = 0.8333）。

## 結果（seed-42, held-out, 60 paired cases × 4 centroids, node cap 300k）

| 偏好 | RL | value_density | best_rule | oracle(下界) | oracle−VD | RL regret vs oracle | VD regret(proven) | proven | RL≥VD cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| count | 0.4018 | 0.4453 | 0.4455 | 0.4485 | +0.0032 | 0.0468 | 0.0048 | 17/60 | 17/60 |
| balanced | 0.3813 | 0.4233 | 0.4236 | 0.4285 | +0.0052 | 0.0471 | 0.0080 | 19/60 | 14/60 |
| gt | 0.3669 | 0.4144 | 0.4148 | 0.4233 | +0.0089 | 0.0564 | 0.0133 | 20/60 | 15/60 |
| risk | 0.4005 | 0.4571 | 0.4573 | 0.4651 | +0.0080 | 0.0645 | 0.0122 | 19/60 | 15/60 |

oracle 為有效下界（僅約 30% case 證明全域最佳，其餘為可實現下界，真實天花板只會更高或相等）。
`best_rule` 幾乎等於 value_density，故 VD 即為最強規則對手。

## 定案結論

**一、value-density 幾乎就是天花板。** 在能證明全域最佳的 case 上，VD 對 oracle 的 regret 僅
0.005–0.013（約 1–3%）。整體 oracle 下界也只比 VD 高 0.003–0.009。也就是說在這個合成環境裡，
偏好 utility 近乎可分離，貪婪的 value-density 已接近最佳，**VD 之上幾乎沒有 headroom**（個別
封港緊迫 case 最多 +0.06，但平均只有 1–2%）。這修正了先前弱 beam 的印象，也給出嚴格定論。

**二、RL 目前落後最強規則一大截，且四種偏好全輸。** RL 對 oracle 的 regret 為 0.047–0.065
（約 11–16%），是 VD regret 的 4–6 倍；RL−VD 為 −0.042 到 −0.057（約 −10% 到 −13%），且在
60 個 case 中只有 14–17 個追平或勝過 VD。（更正先前結論：當 VD 正確地對每個偏好各自最佳化
後，RL 連 GT 偏好也輸給 VD。）

**三、Goal 2（接近／打平）可行性定論：**

- 「**打平 value-density**」是**良置問題且原則上可達**——天花板在 VD 之上，VD 又在 RL 之上，
  RL 要追的是一個明確存在、約 0.045 的 utility 差距，不是雜訊。但目前**尚未達成**：單 seed、
  100k transitions 的 frozen-router pilot 明顯欠訓練。要打平需要的是**訓練層面的改動**
  （VD warm-start／behavior cloning + 更長 steps + 多 seed），不是再調 router。
- 「**勝過 value-density**」在現有合成環境**幾乎不可能**：平均只有 1–3% headroom，且集中在少數
  封港緊迫 case。RL 唯一有結構性優勢的地方就是這些緊迫 case（oracle−VD 最大），可作為
  課程／取樣重點，但整體要穩定超越 VD 不切實際。
- **RL 真正的價值主張因此不是 utility 超越，而是連續偏好介面 + 安全 gate + 單一 checkpoint 的
  Pareto 集**；utility 目標應正式定為「在 CI 內追平 VD」。

**四、根本限制**：VD 之所以近最佳，是因為目前 utility 近乎可分離、耦合只來自拖船／入口／封港。
RL 相對貪婪的優勢要能顯現，需要環境具備**非短視結構**——也就是 CLAUDE.md 鐵律提到、但資料源
尚未到位的真實靠泊／離泊 service time（`ServiceTimeSource`）。在合成環境上再訓練，天花板都被
貪婪壓著。這是 Goal 2 的硬邊界。

### 交棒 Phase 11 後續

oracle 已完成並定案「打平 = 良置、勝過 = 幾乎不可能」。下一步（若要追平 VD）：value-density
warm-start + 較長訓練 + 多 seed，並以 per-case regret vs oracle 當驗收；緊迫封港子集為 RL
唯一 headroom，優先加權。詳見 [PHASE11_13_ROADMAP.md](PHASE11_13_ROADMAP.md)。

## 執行

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m typhoon.evaluate_oracle \
  --db /path/to/ua1008l.sqlite --node-cap 300000 \
  --model typhoon/models/soft_moe_router_ppo/seed-42/final.pt
~~~
