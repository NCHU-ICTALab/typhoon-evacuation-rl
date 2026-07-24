# Phase 6：Custom decomposed PPO fixed-profile diagnostic

## 目的

Phase 5 的完整 actor experts 仍出現循環錯位。本階段在加入 soft MoE 前，先判斷錯位是否
來自多偏好共享：以同一套 custom `DecomposedActorCritic`、vector GAE、late scalarization
與 optimizer，分別訓練 count、balanced、GT、risk 四個完全獨立模型。

四顆模型不共享 extractor、critic、optimizer、advantage normalization 或 rollout buffer。
如果仍無法恢復舊 specialist 已證明可達的 GT、risk 對角，soft MoE 就沒有正確 experts 可混合。

## 受控設定

- 每模型固定一個 preference profile；
- 每模型 100,000 transitions，seed 42；
- paired deterministic scenario stream 與相同 24 個 train dates；
- rollout 512、batch 64、10 epochs；batch 64 對應 routed model 每個 profile 在混合 batch 中
  約 64 筆的有效更新粒度；
- shared actor mode、三頭 critic、PCGrad 關閉；
- 30 艘、31 個動作與 hard action mask 不變；
- 四模型合計 400,000 transitions，輸出到 `decomposed_ppo_fixed_profiles/`。

## 命令

~~~bash
for profile in count balanced gt risk; do
  .venv/bin/python -m typhoon.train_decomposed_ppo \
    --db data/ua1008l.sqlite \
    --steps-per-profile 100000 \
    --seed 42 \
    --batch-size 64 \
    --fixed-profile "$profile" \
    --skip-evaluation
done

.venv/bin/python -m typhoon.evaluate_decomposed_specialists \
  --db data/ua1008l.sqlite \
  --seed 42
~~~

## Soft MoE gate

只有在 custom fixed-profile 至少恢復 GT、risk 的對角最佳，才值得把 experts 放入 soft
router。若 gate 失敗，下一步應修正 actor surrogate／advantage scalarization，而不是增加 router。
