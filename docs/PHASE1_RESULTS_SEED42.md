# Phase 1 Conditioned v2：Seed 42 結果

## 改造

- 四個固定 profile 環境同步提供 transitions，每個 profile 精確占 rollout 的 25%；
- 各 profile 依 episode index 使用同一套 deterministic scenario stream；
- preference 先編碼為 32 維 embedding；
- FiLM 依 preference 對每艘船的 32 維 embedding 做縮放與位移；
- 單一 MaskablePPO 模型，四偏好各 100,000 transitions，總計 400,000；
- 封港、入口與拖船限制仍由 action mask 強制。

注意：v2 同時改變資料配置、網路表示與總資料量（baseline 為 100,000 total），因此本次
不能把所有差異單獨歸因於 FiLM。後續 ablation 應分別比較 balanced-only 與 gating-only。

## 驗收

| 項目 | Baseline | v2 | 判定 |
|---|---:|---:|---|
| Safety violations | 0 | 0 | 通過 |
| Rejected actions | 0 | 0 | 通過 |
| 有偏好 action 差異的 paired cases | 20/60 | 55/60 | 明顯改善 |
| 四種 action trace 全不同 | 0/60 | 35/60 | 明顯改善 |
| Cross-utility 對角最佳 | 0/4 | 1/4（count） | 尚未通過 |

v2 已解決大部分 representation collapse：只有 5/60 案例仍四偏好完全相同。但不同排程
未必朝正確效用方向移動，因此「行為不同」不能單獨當作成功。

## 各偏好效用

| Preference | Baseline | v2 | Specialist | v2 相對 specialist regret |
|---|---:|---:|---:|---:|
| count | 0.3998 | 0.4092 | 0.4031 | -1.5%（v2 較高） |
| balanced | 0.3827 | 0.3863 | 0.3895 | 0.8% |
| GT | 0.3573 | 0.3480 | 0.3683 | 5.5% |
| risk | 0.3878 | 0.3929 | 0.4048 | 2.9% |

相較 baseline，count、balanced、risk utility 改善，GT 下降。v2 的 cross-utility matrix 中，
balanced、GT、risk 三列仍由 count 輸出取得較高分，顯示共享 scalar critic／policy update
仍有梯度干擾；偏好已能改變表示，但 credit assignment 尚未正確分解。

## 決策

暫不擴到多 seed，也不只增加 v2 steps。下一個實驗應處理 objective gradient interference：

1. vector rollout buffer 保存三目標 reward；
2. 三個 value heads；
3. 每個 objective 分別計算 GAE 與 clipped PPO loss；
4. objective loss 穩定後才套用 preference weights；
5. 保留 action mask 與目前 preference gating；
6. 先用 seed 42 與同一 held-out protocol 驗證，再擴 seeds 43、44。

這對應計畫中的 D3PO-lite；D3PO 仍是預印本，因此 specialist 與 value-density 必須繼續
保留為對照。

完整輸出位於被 Git 忽略的：

- `typhoon/models/conditioned_v2/seed-42/evaluation.json`
- `typhoon/models/conditioned_v2/seed-42/comparison.json`
- `typhoon/models/conditioned_v2/seed-42/training_manifest.json`
