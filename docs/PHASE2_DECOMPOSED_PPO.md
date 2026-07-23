# Phase 2：Decomposed Maskable PPO（D3PO-lite）

## 為何採用

Phase 1 已讓偏好改變 action trace（55/60 paired cases），但只有 count 達成 cross-utility
對角最佳；GT utility 反而下降。問題已不是 policy 看不到 preference，而是三個 objective 在
進入 PPO 前被加成一個 scalar advantage，導致共享 critic 的 credit assignment 與 actor
梯度互相干擾。

本階段不刻意放大梯度，也不強迫排程彼此不同。目標是先完整保存每個 objective 的 return、
value、advantage 與 clipped surrogate，再於 actor loss 的最後一步依 preference 加權。

## 使用技術與理由

| 技術 | 理由 |
|---|---|
| Balanced four-profile rollout | 每個 profile 精確貢獻 25% transitions，避免抽樣偏差 |
| Deterministic scenario stream | 各 profile 使用相同情境分布，差異只來自 preference |
| Preference encoder + FiLM | Phase 1 已證明能降低 representation collapse，予以保留 |
| Vector rollout buffer | 不丟失 count、GT、risk 的個別 reward |
| Three-head critic | 分別估計目前 conditioned policy 下的三目標長期 value |
| Per-objective GAE | 分別完成 credit assignment，避免先混成 scalar TD error |
| Per-objective PPO clipping | 每個 objective 先獨立穩定更新，再進行偏好加權 |
| Late preference weighting | actor 最後才依每筆 transition 的 preference 合成 loss |
| Per-objective advantage normalization | 避免某目標只因數值尺度較大而支配更新 |
| Equal critic loss | 即使某回合權重低，對應 critic 仍能持續學習 |
| Hard action mask | 安全限制不與任何 objective 交換 |
| Gradient cosine diagnostics | 量測目標方向衝突，決定後續是否需要 PCGrad |

## 刻意暫緩的技術

- Diversity regularizer：Phase 1 已有 55/60 行為分離；目前應修正方向而非強迫更不同。
- PCGrad／CAGrad：先量測分解後的 gradient cosine；若仍頻繁為負再加入。
- Envelope Q-learning：保留為另一套離散 MORL benchmark，不混入 PPO optimizer。

## 演算法契約

對 objective \(j\in\{C,G,R\}\)：

\[
\delta_t^j=r_t^j+\gamma(1-d_t)V_j(s_{t+1},w)-V_j(s_t,w)
\]

\[
A_t^j=\delta_t^j+\gamma\lambda(1-d_t)A_{t+1}^j
\]

每個 objective 先進行 PPO clipping：

\[
S_t^j=\min(\rho_t A_t^j,\operatorname{clip}(\rho_t)A_t^j)
\]

最後才組合 actor loss：

\[
L_{actor}=-E_t\left[\sum_j w_t^j S_t^j\right]
\]

critic loss 對三個 head 等權平均。所有 reward、value、advantage、return shape 固定為 `(3,)`。

## Seed 42 驗收標準

1. safety violations 與 rejected actions 維持 0；
2. 偏好 action 差異案例不低於 Phase 1 的 55/60；
3. cross-utility 對角最佳由 1/4 提升；
4. GT utility 高於 Phase 1 的 0.3480；
5. risk specialist regret低於 Phase 1 的 2.9%；
6. 保存 count-GT、count-risk、GT-risk 的平均 gradient cosine；
7. 若未通過，不直接增加 steps，先依 cosine 決定是否採用 gradient surgery。

這是一個研究用 D3PO-lite 實作，不宣稱等同預印本的完整演算法或結果。
