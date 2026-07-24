# Phase 8：Preference-scalarized critic loss

## 目的

Phase 7 已修正 actor surrogate，成功恢復 GT，但 risk 仍未對角最佳。固定偏好下，vector
GAE 的偏好加權與 scalar GAE 具有線性等價性；剩餘的主要差異之一，是 custom critic 對
三個 value heads 等權訓練，而舊 SB3 specialist 只擬合當前偏好的 scalar value。

Phase 8 保留三頭 critic 與 vector returns，但將 critic loss 改為：

~~~text
V_scalar = Σ_j w_j V_j
R_scalar = Σ_j w_j R_j
L_scalar = MSE(V_scalar, R_scalar)
L_vector = MSE(V, R)
L_critic = L_scalar + 0.1 × L_vector
~~~

`L_scalar` 將主要 critic 容量對準當前偏好；0.1 的 `L_vector` auxiliary loss 避免三個 value
heads 只學到不可辨識的任意分解。Actor 維持 Phase 7 的 preference-first scalar advantage
與單次 PPO clipping。

## 受控條件

- 每 profile 100,000 transitions，seed 42；
- rollout 512、batch 64、epochs 10；
- shared actor、無 PCGrad、無 routing；
- `vector_value_aux_coef=0.1`；
- 先驗收 GT/risk，通過後再補 count/balanced；
- 輸出到新目錄，不覆寫 Phase 7。

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
    --critic-mode preference_scalar_aux \
    --vector-value-aux-coef 0.1 \
    --skip-evaluation
done

.venv/bin/python -m typhoon.evaluate_decomposed_specialists \
  --db data/ua1008l.sqlite \
  --seed 42 \
  --model-root typhoon/models/decomposed_ppo_preference_first_scalar_critic_fixed_profiles \
  --algorithm custom_decomposed_ppo_preference_first_scalar_critic_fixed_profile
~~~
