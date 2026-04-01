from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analisis de resultados ACO")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/sim_results/aco_experiments_full.csv"),
        help="CSV de entrada con corridas",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/sim_results/analysis"),
        help="Directorio de salida",
    )
    return p.parse_args()


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def iqr(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(0.75) - s.quantile(0.25))


def build_summary(ok: pd.DataFrame) -> pd.DataFrame:
    grp = ok.groupby(["scenario", "weight", "config"], dropna=False)
    rows: List[Dict[str, float]] = []
    for (scenario, weight, config), g in grp:
        row = {
            "scenario": scenario,
            "weight": weight,
            "config": config,
            "runs": int(len(g)),
            "objective_cost_median": float(g["objective_cost"].median()),
            "objective_cost_iqr": iqr(g["objective_cost"]),
            "distance_total_median": float(g["distance_total"].median()),
            "distance_total_iqr": iqr(g["distance_total"]),
            "penalty_total_median": float(g["penalty_total"].median()),
            "penalty_total_iqr": iqr(g["penalty_total"]),
            "exec_s_median": float(g["exec_s"].median()),
            "exec_s_iqr": iqr(g["exec_s"]),
            "pct_on_time_median": float(g["pct_on_time"].median()),
            "pct_on_time_iqr": iqr(g["pct_on_time"]),
            "penalty_window_median": float(g["penalty_window"].median()),
            "penalty_capacity_median": float(g["penalty_capacity"].median()),
            "penalty_time_median": float(g["penalty_time"].median()),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary = summary.sort_values(["scenario", "weight", "objective_cost_median", "objective_cost_iqr"])
    summary["rank_in_scenario_weight"] = (
        summary.groupby(["scenario", "weight"])["objective_cost_median"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return summary


def plot_scale_metric_by_scenario_weight(ok: pd.DataFrame, out_dir: Path, metric: str) -> None:
    for (scenario, weight), g in ok.groupby(["scenario", "weight"], dropna=False):
        cfgs = list(g["config"].dropna().unique())
        data = [g.loc[g["config"] == c, metric].dropna().tolist() for c in cfgs]
        labels = [c for c, vals in zip(cfgs, data) if len(vals) > 0]
        data = [vals for vals in data if len(vals) > 0]
        if not data:
            continue

        plt.figure(figsize=(10, 5))
        plt.boxplot(data, tick_labels=labels)
        plt.title(f"Boxplot - {metric} | scenario={scenario} | weight={weight}")
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fname = f"boxplot_{metric}_scenario_{scenario}_weight_{weight}.png".replace("/", "_")
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()


def plot_global_comparable_metrics(ok: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["exec_s", "pct_on_time"]
    cfgs = list(ok["config"].dropna().unique())

    for metric in metrics:
        data = [ok.loc[ok["config"] == c, metric].dropna().tolist() for c in cfgs]
        labels = [c for c, vals in zip(cfgs, data) if len(vals) > 0]
        data = [vals for vals in data if len(vals) > 0]
        if not data:
            continue

        plt.figure(figsize=(11, 5))
        plt.boxplot(data, tick_labels=labels)
        plt.title(f"Boxplot - {metric} (comparable entre pesos)")
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"boxplot_{metric}_all_configs.png", dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"No existe archivo de entrada: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "status" not in df.columns:
        raise SystemExit("El CSV no contiene columna status")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise SystemExit("No hay corridas OK para analizar")

    numeric_cols = [
        "objective_cost",
        "distance_total",
        "penalty_total",
        "penalty_window",
        "penalty_capacity",
        "penalty_time",
        "total_time",
        "pct_on_time",
        "exec_s",
    ]
    ok = ensure_numeric(ok, numeric_cols)

    summary = build_summary(ok)

    summary_csv = args.out_dir / "summary_by_scenario_weight.csv"
    summary_md = args.out_dir / "summary_by_scenario_weight.md"
    summary.to_csv(summary_csv, index=False)

    for metric in ["objective_cost", "distance_total", "penalty_total"]:
        plot_scale_metric_by_scenario_weight(ok, args.out_dir, metric)

    plot_global_comparable_metrics(ok, args.out_dir)

    print("Analisis completado")
    print(f"- Resumen CSV: {summary_csv}")
    print(f"- Resumen MD:  {summary_md}")
    print(f"- Boxplots:    {args.out_dir}")


if __name__ == "__main__":
    main()
