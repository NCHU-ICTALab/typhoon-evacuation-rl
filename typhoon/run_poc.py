"""Command-line entry point for the offline typhoon evacuation PoC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_scenario import DEFAULT_DB, build_scenario, write_scenario
from .planner import compare_policies


OUT_DIR = Path(__file__).resolve().parent / "out"


def _markdown(report: dict) -> str:
    meta = report["scenario_metadata"]
    fcfs = report["policies"]["fcfs"]["kpi"]
    risk = report["policies"]["risk_aware"]["kpi"]
    delta = report["delta_risk_aware_minus_fcfs"]
    classes = meta["data_classes"]
    return f"""# 颱風封港前撤離排序 PoC 報告

- 歷史船舶樣本日期：{meta['query_date']}
- 船舶資料：{classes['vessels']}
- 天氣資料：{classes['weather']}
- 資源資料：{classes['resources']}
- 風險分數：{classes['risk_points']}

| KPI | FCFS | Risk-aware | 差值 |
|---|---:|---:|---:|
| 撤離艘數 | {fcfs['evacuated_count']} | {risk['evacuated_count']} | {delta['evacuated_count']:+d} |
| 撤離 GT | {fcfs['evacuated_gt']:.0f} | {risk['evacuated_gt']:.0f} | {delta['evacuated_gt']:+.0f} |
| 撤離風險點數 | {fcfs['evacuated_risk_points']} | {risk['evacuated_risk_points']} | {delta['evacuated_risk_points']:+d} |
| 未撤離艘數 | {fcfs['remaining_count']} | {risk['remaining_count']} | {delta['remaining_count']:+d} |

> 本報告使用合成風力、資源容量與風險 proxy，只能證明程式和決策流程可運作，
> 不能宣稱為真實颱風成效或官方撤離優先序。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--date", default=None)
    parser.add_argument("--closure-hour", type=float, default=8.0)
    parser.add_argument("--horizon-hour", type=float, default=24.0)
    parser.add_argument("--tug-capacity", type=int, default=8)
    parser.add_argument("--max-vessels", type=int, default=30)
    parser.add_argument("--demand-compression", type=float, default=2.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    scenario = build_scenario(
        db_path=args.db,
        query_date=args.date,
        closure_hour=args.closure_hour,
        horizon_h=args.horizon_hour,
        tug_capacity=args.tug_capacity,
        max_vessels=args.max_vessels,
        demand_compression=args.demand_compression,
    )
    report = compare_policies(scenario)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_scenario(scenario, args.out_dir / "latest_scenario.json")
    (args.out_dir / "latest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = _markdown(report)
    (args.out_dir / "latest_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"輸出已覆寫至 {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
