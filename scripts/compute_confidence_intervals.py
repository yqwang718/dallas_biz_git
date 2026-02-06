"""
Compute confidence intervals for estimators from JSON result files.

Usage:
    python scripts/compute_confidence_intervals.py --data-root data/estimators/biz --export-csv
    python scripts/compute_confidence_intervals.py --alpha 0.01  # for 99% CI
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy import stats


def load_estimator(path: str) -> dict:
    """Load estimator JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_ci(
    estimate: float,
    se: float,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute confidence interval.
    
    Args:
        estimate: Point estimate (coefficient)
        se: Standard error
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = estimate - z * se
    ci_upper = estimate + z * se
    return ci_lower, ci_upper


def significance_stars(estimate: float, se: float) -> str:
    """Return significance stars based on z-test against zero."""
    if se == 0:
        return ""
    z = abs(estimate / se)
    p = 2 * (1 - stats.norm.cdf(z))
    
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def extract_ci_from_results(
    estimator: dict,
    crime_type: str,
    alpha: float = 0.05
) -> List[dict]:
    """
    Extract confidence intervals from a single crime type result.
    
    Returns:
        List of dicts with variable, estimate, SE, CI, and significance
    """
    results = estimator["results"][crime_type]
    est = results["estimators"]
    se = results["standard_errors"]
    
    rows = []
    
    # Distance
    if est.get("distance") is not None:
        ci_lo, ci_hi = compute_ci(est["distance"], se["distance"], alpha)
        rows.append({
            "variable": "distance",
            "estimate": est["distance"],
            "se": se["distance"],
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "sig": significance_stars(est["distance"], se["distance"])
        })
    
    # Race
    if est.get("race") is not None:
        ci_lo, ci_hi = compute_ci(est["race"], se["race"], alpha)
        rows.append({
            "variable": "race",
            "estimate": est["race"],
            "se": se["race"],
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "sig": significance_stars(est["race"], se["race"])
        })
    
    # Income
    if est.get("income") is not None:
        ci_lo, ci_hi = compute_ci(est["income"], se["income"], alpha)
        rows.append({
            "variable": "income",
            "estimate": est["income"],
            "se": se["income"],
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "sig": significance_stars(est["income"], se["income"])
        })
    
    # Features
    if "features" in est:
        for name, value in est["features"].items():
            if value is not None:
                se_val = se["features"][name]
                ci_lo, ci_hi = compute_ci(value, se_val, alpha)
                rows.append({
                    "variable": name,
                    "estimate": value,
                    "se": se_val,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "sig": significance_stars(value, se_val)
                })
    
    return rows


def process_single_file(
    filepath: Path,
    alpha: float = 0.05,
    crime_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Process a single estimator JSON file and extract CIs for all crime types.
    
    Args:
        filepath: Path to JSON file
        alpha: Significance level for CI
        crime_types: List of crime types to process (None = all)
    
    Returns:
        DataFrame with all CIs
    """
    estimator = load_estimator(str(filepath))
    
    if crime_types is None:
        crime_types = list(estimator["results"].keys())
    
    all_rows = []
    for crime_type in crime_types:
        if crime_type not in estimator["results"]:
            continue
        
        rows = extract_ci_from_results(estimator, crime_type, alpha)
        for row in rows:
            row["crime_type"] = crime_type
        all_rows.extend(rows)
    
    df = pd.DataFrame(all_rows)
    if not df.empty:
        # Reorder columns
        df = df[["crime_type", "variable", "estimate", "se", "ci_lower", "ci_upper", "sig"]]
    
    return df


def process_directory(
    data_root: str = "data/estimators/biz",
    alpha: float = 0.05,
    crime_types: Optional[List[str]] = None,
    print_results: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Process all JSON files in a directory.
    
    Returns:
        Dictionary mapping filename (without extension) to DataFrame
    """
    data_root = Path(data_root)
    json_files = sorted(data_root.glob("*.json"))
    
    results = {}
    
    for filepath in json_files:
        name = filepath.stem
        
        try:
            df = process_single_file(filepath, alpha, crime_types)
            results[name] = df
            
            if print_results:
                print(f"\n{'='*80}")
                print(f"File: {name}.json")
                print(f"{'='*80}")
                
                for crime_type in df["crime_type"].unique():
                    ct_df = df[df["crime_type"] == crime_type]
                    print(f"\n  {crime_type}:")
                    print(f"  {'Variable':<35} {'Est':>10} {'SE':>10} {'95% CI':>25} {'Sig':>5}")
                    print(f"  {'-'*85}")
                    
                    for _, row in ct_df.iterrows():
                        ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                        print(f"  {row['variable']:<35} {row['estimate']:>10.4f} "
                              f"{row['se']:>10.4f} {ci_str:>25} {row['sig']:>5}")
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    return results


def export_to_csv(
    results: Dict[str, pd.DataFrame],
    output_path: str = "data/tables/confidence_intervals.csv"
):
    """Export all results to a single CSV file."""
    dfs = []
    for name, df in results.items():
        df_copy = df.copy()
        df_copy.insert(0, "model", name)
        dfs.append(df_copy)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\nResults exported to {output_path}")
    return combined


def export_to_latex(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "data/tables",
    crime_type: str = "all_crime_types"
):
    """Export results to LaTeX tables (one per model, filtered by crime type)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in results.items():
        # Filter to specified crime type
        ct_df = df[df["crime_type"] == crime_type].copy()
        if ct_df.empty:
            continue
        
        # Format for LaTeX
        latex_df = ct_df[["variable", "estimate", "se", "ci_lower", "ci_upper", "sig"]].copy()
        latex_df.columns = ["Variable", r"$\beta$", "SE", "CI Lower", "CI Upper", ""]
        
        # Format numbers
        for col in [r"$\beta$", "SE", "CI Lower", "CI Upper"]:
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.3f}")
        
        latex_str = latex_df.to_latex(
            index=False,
            escape=False,
            column_format="l" + "r" * 5,
            caption=f"Coefficient estimates with 95\\% confidence intervals: {name}",
            label=f"tab:ci_{name}"
        )
        
        output_path = output_dir / f"ci_{name}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        print(f"LaTeX table exported to {output_path}")


def create_summary_table(
    results: Dict[str, pd.DataFrame],
    crime_type: str = "all_crime_types"
) -> pd.DataFrame:
    """
    Create a summary table comparing coefficients across models for a specific crime type.
    Columns: variable, then one column per model with estimate and CI.
    """
    summary_rows = []
    
    # Get all unique variables across models
    all_vars = set()
    for df in results.values():
        ct_df = df[df["crime_type"] == crime_type]
        all_vars.update(ct_df["variable"].tolist())
    
    # For each variable, collect estimates from all models
    for var in sorted(all_vars):
        row = {"variable": var}
        for model_name, df in results.items():
            ct_df = df[(df["crime_type"] == crime_type) & (df["variable"] == var)]
            if not ct_df.empty:
                r = ct_df.iloc[0]
                row[f"{model_name}_est"] = r["estimate"]
                row[f"{model_name}_ci"] = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
                row[f"{model_name}_sig"] = r["sig"]
            else:
                row[f"{model_name}_est"] = None
                row[f"{model_name}_ci"] = ""
                row[f"{model_name}_sig"] = ""
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute confidence intervals for estimators")
    parser.add_argument("--data-root", default="data/estimators/biz",
                        help="Root directory for estimator files")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default 0.05 for 95%% CI)")
    parser.add_argument("--crime-type", default=None,
                        help="Specific crime type to analyze (default: all)")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--export-latex", action="store_true",
                        help="Export results to LaTeX tables")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")
    
    args = parser.parse_args()
    
    crime_types = [args.crime_type] if args.crime_type else None
    
    # Process all files
    results = process_directory(
        args.data_root,
        alpha=args.alpha,
        crime_types=crime_types,
        print_results=not args.quiet
    )
    
    # Export if requested
    if args.export_csv:
        export_to_csv(results)
    
    if args.export_latex:
        export_to_latex(results, crime_type=args.crime_type or "all_crime_types")
    
    # Print summary
    ci_pct = int((1 - args.alpha) * 100)
    print(f"\n{'='*80}")
    print(f"SUMMARY: {ci_pct}% Confidence Intervals")
    print(f"{'='*80}")
    print(f"Processed {len(results)} model files")
    
    for name, df in results.items():
        n_sig = (df["sig"] != "").sum()
        n_total = len(df)
        print(f"  {name}: {n_sig}/{n_total} coefficients significant at p < {args.alpha}")

