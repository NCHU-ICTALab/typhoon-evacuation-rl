# 偏好學習實驗規約

## 不可變安全條件

- 封港後不得開始或完成出港動作；
- 不得超過入口同時容量；
- 不得超過拖船容量；
- `safety_violations == 0` 且 `rejected_actions == 0`。

## 配對案例

比較四種偏好時，以下欄位必須完全相同：歷史日期、closure hour、tug capacity、demand
compression、jitter 開關與 RNG seed。只允許 preference weights 不同。

## 報告指標

1. 原始目標：撤離 count、GT、risk points；
2. 每一偏好的 weighted utility；
3. cross-utility matrix；
4. 同案例四偏好的 unique action traces 與 unique schedules；
5. conditioned policy 對對應 specialist 的 utility regret；
6. Pareto 非支配解數、hypervolume、epsilon indicator 與 sparsity；
7. 多 seed 平均、95% CI 與 paired bootstrap。

只有在同一狀態存在兩個以上可行且互有取捨的動作時，才評估 preference sensitivity。
沒有取捨的狀態採取相同動作是正確結果。

## 建議正式訓練矩陣

- 訓練 seed：至少 3，建議 5；
- profile：count、balanced、GT、risk、conditioned；
- held-out dates：依時間切分最後 5 日；
- steps：先以 100k 做診斷，再依 learning curve 決定，不直接預設 1M；
- 每次執行保存程式 commit SHA、參數與 manifest，但不將模型檔推上 GitHub。
