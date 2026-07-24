# Phase 3：Shared-policy PCGrad 消融

## 目的

Phase 2 的 post-hoc 診斷顯示 count–GT 與 GT–risk 分別有 38% 與 40% 的負 cosine，
因此本階段只回答一個問題：在其他設定固定時，對 shared actor/extractor 的 objective policy
gradient 套用 PCGrad，是否能改善 preference routing 與 cross-utility 對角最佳。

這不是新模型架構，也不調整 30 艘船、情境資料、reward、critic、action mask 或訓練量。

## 實作契約

每個 objective 先完成 per-objective PPO clipping，並保留 transition preference：

\[
L_j=-E_t[w_t^j S_t^j],\qquad g_j=\nabla_{\theta_{shared}}L_j
\]

若兩個 objective 的 dot product 為負，才進行投影：

\[
g_i \leftarrow g_i-\frac{g_i^Tg_j}{\lVert g_j\rVert^2}g_j
\]

三組投影後 policy gradient 相加，再加上原本的 equal-weight vector critic loss 與 entropy
gradient。PCGrad 只作用於 `extractor + actor`；三頭 critic 本身不做 gradient surgery。

為維持受控比較：

- minibatch RNG 與 PCGrad 投影順序使用不同 seed；
- 四 profile 繼續各佔 25% transitions；
- 情境、PPO 超參數與 seed 42 Phase 2 相同；
- 模型輸出到獨立的 `decomposed_ppo_pcgrad/`，不覆寫 Phase 2；
- 每個 minibatch 保存 raw cosine、負 cosine 比例與實際投影比例；
- 封港、入口及拖船限制仍由 hard action mask 保證。

## 命令

~~~bash
.venv/bin/python -m typhoon.train_decomposed_ppo \
  --db data/ua1008l.sqlite \
  --steps-per-profile 100000 \
  --seed 42 \
  --gradient-surgery pcgrad
~~~

## 判讀限制

MORL 中的負 gradient 不一定都是有害干擾；有些正是 count、GT、risk 真實取捨的表現。
因此 PCGrad 是否成功不能只看 cosine 或投影比例，必須以 held-out utility、specialist regret、
cross-utility 對角最佳及安全 KPI 共同判斷。
