# Phase 4 Hard preference heads：Seed 42 結果

## 設定

- 固定 30 艘、31 個離散動作；
- shared preference-gated extractor + actor trunk；
- count、balanced、GT、risk 四個相同初始化的 output heads；
- 共享三頭 critic，PCGrad 關閉；
- 四 profile 各 100,000 transitions，總計 400,000；
- 五個 held-out 日期、每 profile 60 個完全配對案例。

技術契約見 `PHASE4_HARD_HEADS.md`。本結果仍是 seed 42 單次結構消融。

## 驗收摘要

| 項目 | Phase 2 | PCGrad | Hard heads |
|---|---:|---:|---:|
| Safety violations / rejected | 0 / 0 | 0 / 0 | 0 / 0 |
| 有偏好 action 差異 | 55/60 | 57/60 | **59/60** |
| 四種 action trace 全不同 | 33/60 | 36/60 | **52/60** |
| Cross-utility 對角最佳 | 1/4 | 0/4 | 1/4（count） |

output heads 幾乎完全解決「四個偏好產生相同行為」，但沒有解決「不同偏好選到正確
Pareto region」。因此行為分離是必要條件，不是 preference learning 成功的充分條件。

## Utility 與 specialist regret

| Preference | Phase 2 utility | Hard heads | Delta | Specialist | Hard-head regret |
|---|---:|---:|---:|---:|---:|
| count | 0.3929 | **0.4068** | +0.0139 | 0.4031 | -0.9% |
| balanced | 0.3821 | 0.3814 | -0.0007 | 0.3895 | 2.1% |
| GT | 0.3633 | 0.3629 | -0.0004 | 0.3683 | 1.5% |
| risk | 0.3913 | 0.3875 | -0.0038 | 0.4048 | 4.3% |

count head 超過原 count specialist，但 balanced、GT、risk 沒有受益，risk regret 反而增加。

## Cross-utility matrix

列是評分偏好，欄是產生排程的 head：

| 評分偏好 \ Head | count | balanced | GT | risk |
|---|---:|---:|---:|---:|
| count | **0.406770** | 0.395000 | 0.402017 | 0.392155 |
| balanced | **0.394674** | 0.381419 | 0.386148 | 0.382568 |
| GT | **0.371134** | 0.358861 | 0.362903 | 0.368050 |
| risk | **0.406118** | 0.390396 | 0.393525 | 0.387498 |

只有 count 對角最佳。count head 在其他三種評分下也都最高，顯示其排程不是單純偏向
count，而是整體優於另外三個尚未學好的 heads。

## 三目標成分

由 cross-utility 的四組線性權重反解各 head 的平均 normalized completion：

| Producing head | Count | GT | Risk |
|---|---:|---:|---:|
| count | **0.4167** | 0.3519 | **0.4155** |
| balanced | 0.4061 | 0.3404 | 0.3977 |
| GT | 0.4150 | 0.3439 | 0.3996 |
| risk | 0.4000 | **0.3562** | 0.3915 |

count head 同時 Pareto-dominate balanced 與 GT heads。risk head 學到最高 GT，卻沒有最高
risk；head 確實切換，但目標區域錯位。這不是單靠更多輸出差異可以解決。

## 與 specialist 可達性的關係

四個完全獨立 specialist 的對角最佳只有 2/4：GT 與 risk。count 評分由 balanced specialist
最高，balanced 評分由 risk specialist 最高。因此目前沒有證據保證 4/4 可達；但 hard heads
連已證明可達的 GT、risk 對角都沒有恢復，仍低於合理結構目標。

## 結論與下一步

1. 保留此架構作診斷，不切換前端正式模型；Phase 2 仍是目前 GT/risk 較穩定的 conditioned
   checkpoint。
2. 最後一層分 head 已足以製造不同排程，但 shared actor trunk 仍可能把四種更新壓到相同
   representation region。
3. 下一個最小結構實驗應改成 shared extractor + 四個完整 actor experts；每個 expert 都有
   自己的 256/128 policy layers，critic 繼續共享，且四個 experts 相同初始化。
4. 下一階段先以 specialist 已證明的 GT、risk 對角與 regret 為主要門檻；不盲目要求 4/4。
5. 若完整 experts 仍無法恢復 GT/risk，應停止擴 actor，回查 preference-weighted policy
   objective、reward normalization 與 scenario trade-off，而不是直接上 soft MoE。

完整輸出位於被 Git 忽略的：

- `typhoon/models/decomposed_ppo_hard_heads/seed-42/final.pt`
- `typhoon/models/decomposed_ppo_hard_heads/seed-42/training_manifest.json`
- `typhoon/models/decomposed_ppo_hard_heads/seed-42/evaluation.json`
- `typhoon/models/decomposed_ppo_hard_heads/seed-42/comparison.json`
