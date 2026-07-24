# Phase 3 PCGrad：Seed 42 結果

## 設定

- 固定 30 艘船、31 個離散動作；
- 單一 preference-conditioned actor + 三頭 critic；
- 四 profile 各 100,000 transitions，總計 400,000；
- shared actor/extractor 的三組 preference-weighted objective gradient 使用 PCGrad；
- critic、entropy、PPO clipping、情境流、seed 與 Phase 2 相同；
- 五個 held-out 日期、每 profile 60 個完全配對案例。

技術契約見 `PHASE3_PCGRAD.md`。本結果仍是 seed 42 單次受控消融。

## 驗收摘要

| 項目 | Phase 2 | PCGrad | 判定 |
|---|---:|---:|---|
| Safety violations / rejected | 0 / 0 | 0 / 0 | 通過 |
| 有偏好 action 差異 | 55/60 | 57/60 | +2 |
| 四種 action trace 全不同 | 33/60 | 36/60 | +3 |
| Cross-utility 對角最佳 | 1/4 | 0/4 | 退步 |

PCGrad 增加了排程差異，但差異沒有正確對應要求的偏好，因此不能把「更不一樣」視為
Pareto preference learning 改善。

## Utility 與 specialist regret

| Preference | Phase 2 utility | PCGrad utility | Delta | Phase 2 regret | PCGrad regret |
|---|---:|---:|---:|---:|---:|
| count | 0.3929 | 0.3916 | -0.0013 | 2.5% | 2.8% |
| balanced | 0.3821 | 0.3796 | -0.0025 | 1.9% | 2.5% |
| GT | 0.3633 | 0.3486 | -0.0147 | 1.4% | 5.4% |
| risk | 0.3913 | 0.3892 | -0.0021 | 3.3% | 3.9% |

GT 的退步最大，幾乎回到 Phase 1 的 0.3480；這表示 decomposed critic 所修復的 GT credit
assignment，沒有因 PCGrad 得到進一步改善，反而被 shared-policy 投影抵銷。

## Cross-utility matrix

列是評分使用的偏好，欄是產生排程時輸入模型的偏好：

| 評分偏好 \ 產生偏好 | count | balanced | GT | risk |
|---|---:|---:|---:|---:|
| count | 0.391572 | **0.399070** | 0.395139 | 0.398793 |
| balanced | 0.374480 | 0.379600 | 0.377656 | **0.379665** |
| GT | 0.348743 | 0.349915 | 0.348561 | **0.350992** |
| risk | 0.383125 | **0.389817** | 0.389266 | 0.389208 |

四列最佳值都不在對角線。特別是 risk 輸入在 count、balanced、GT 三種評分下都很強，
顯示目前仍是 preference routing 錯位，而不是單純缺少輸出差異。

## 訓練期間 gradient 診斷

以下是每個 minibatch、preference-weighted objective actor gradient 在投影前的統計：

| Pair | Mean raw cosine | 負 cosine 比例 |
|---|---:|---:|
| count–GT | 0.1174 | 37.59% |
| count–risk | 0.4569 | 7.71% |
| GT–risk | 0.1214 | 37.06% |

PCGrad 的 directed comparison 實際投影比例為 **26.43%**。這證明程式確實頻繁執行投影，
但投影沒有轉化為 held-out utility 改善。

## 結論與下一步

1. 不以 PCGrad checkpoint 取代 Phase 2；安全雖維持，但 utility、regret 與對角最佳退步。
2. 負 cosine 在本環境不全是有害 noise；它也包含 MORL 本來就必須保留的目標取捨。
3. 保留 decomposed critic 與 late scalarization，停止繼續調大 PCGrad 投影強度。
4. 下一個最小實驗改為 actor routing ablation：shared trunk + preference-specific heads，或
   soft mixture-of-experts；目標是讓偏好選擇正確 policy region，而不是消除所有衝突。
5. 在結構消融出現 cross-utility 改善前，不擴到多 seed，也不切換前端正式模型。

完整輸出位於被 Git 忽略的：

- `typhoon/models/decomposed_ppo_pcgrad/seed-42/final.pt`
- `typhoon/models/decomposed_ppo_pcgrad/seed-42/training_manifest.json`
- `typhoon/models/decomposed_ppo_pcgrad/seed-42/evaluation.json`
- `typhoon/models/decomposed_ppo_pcgrad/seed-42/comparison.json`
