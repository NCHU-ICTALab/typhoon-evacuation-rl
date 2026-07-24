# Phase 7：Preference-first scalar PPO surrogate

## 目的

Phase 6 已確認四個完全獨立的 custom fixed-profile models 仍無法恢復 GT、risk 的對角
最佳。Phase 7 只改 actor surrogate 的運算順序，測試問題是否來自逐目標 clipping 的
非線性。

## 更新順序

Critic 與 rollout 維持向量形式：

~~~text
vector reward → three-head critic → vector GAE / returns
~~~

Actor 改為：

~~~text
vector advantage
→ 依每筆 transition 的 preference 做加權和
→ 對 scalar advantage 做一次 normalization
→ 做一次 PPO clipping
→ actor loss
~~~

公式為：

~~~text
A_scalar(t) = Σ_j w(t,j) A(t,j)
A_norm(t) = normalize(A_scalar(t))
L_actor = -mean(min(ratio(t) A_norm(t), clip(ratio(t)) A_norm(t)))
~~~

Critic 仍以三個 returns 的 MSE 訓練，不把 value heads 合併。為了診斷梯度，程式會將
normalized scalar advantage 拆成三個可加總的 centered contributions；這些 contribution
只用於報表，不會各自進行 clipping。

## 受控條件

- fixed profile、seed、情境順序、rollout、batch、epochs 與 Phase 6 相同；
- 不使用 PCGrad；
- 不使用 hard heads、full experts 或 soft MoE；
- 輸出到 `decomposed_ppo_preference_first_fixed_profiles/`，不覆寫 Phase 6；
- gate 仍是 GT、risk 至少恢復舊 SB3 specialists 的 2/4 對角最佳。

## 執行

~~~bash
for profile in count balanced gt risk; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m typhoon.train_decomposed_ppo \
    --db data/ua1008l.sqlite \
    --steps-per-profile 100000 \
    --seed 42 \
    --batch-size 64 \
    --fixed-profile "$profile" \
    --surrogate-mode preference_first \
    --skip-evaluation
done

.venv/bin/python -m typhoon.evaluate_decomposed_specialists \
  --db data/ua1008l.sqlite \
  --seed 42 \
  --model-root typhoon/models/decomposed_ppo_preference_first_fixed_profiles \
  --algorithm custom_decomposed_ppo_preference_first_fixed_profile
~~~
