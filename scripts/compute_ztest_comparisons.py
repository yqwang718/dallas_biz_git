"""
Compute z-tests for comparing regression coefficients between offender and victim models.

Based on Paternoster et al. (1998):
    z = (b1 - b2) / sqrt(SE1^2 + SE2^2)

Reference: Paternoster, R., Brame, R., Mazerolle, P., & Piquero, A. (1998). 
Using the correct statistical test for the equality of regression coefficients. 
Criminology, 36(4), 859-866.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


def load_estimator(path: str) -> dict:
    """Load estimator JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_coefficients(
    estimator: dict, 
    crime_type: str = "all_crime_types"
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract coefficients and standard errors from estimator results.
    
    Returns:
        Tuple of (coefficients dict, standard_errors dict)
    """
    results = estimator["results"][crime_type]
    
    coefs = {}
    ses = {}
    
    # Distance coefficient
    if results["estimators"]["distance"] is not None:
        coefs["distance"] = results["estimators"]["distance"]
        ses["distance"] = results["standard_errors"]["distance"]
    
    # Race coefficient (if present)
    if results["estimators"].get("race") is not None:
        coefs["race"] = results["estimators"]["race"]
        ses["race"] = results["standard_errors"]["race"]
    
    # Income coefficient (if present)
    if results["estimators"].get("income") is not None:
        coefs["income"] = results["estimators"]["income"]
        ses["income"] = results["standard_errors"]["income"]
    
    # Feature coefficients
    if "features" in results["estimators"]:
        for name, value in results["estimators"]["features"].items():
            if value is not None:
                coefs[name] = value
                ses[name] = results["standard_errors"]["features"][name]
    
    return coefs, ses


def compute_ztest(
    b1: float, se1: float, 
    b2: float, se2: float
) -> Tuple[float, float, str]:
    """
    Compute z-test for difference between two coefficients.
    
    Using Paternoster et al. (1998) Equation 4:
        z = (b1 - b2) / sqrt(SE1^2 + SE2^2)
    
    Returns:
        Tuple of (z-statistic, p-value, significance stars)
    """
    diff = b1 - b2
    se_diff = np.sqrt(se1**2 + se2**2)
    z = diff / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(z)))  # two-tailed
    
    # Significance stars
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = ""
    
    return z, p, sig


def compare_models(
    offender_path: str,
    victim_path: str,
    crime_type: str = "all_crime_types",
    print_results: bool = True
) -> pd.DataFrame:
    """
    Compare offender and victim models using z-tests.
    
    Args:
        offender_path: Path to offender estimator JSON
        victim_path: Path to victim estimator JSON
        crime_type: Crime type to analyze (default: all_crime_types)
        print_results: Whether to print formatted results
    
    Returns:
        DataFrame with comparison results
    """
    # Load estimators
    offender = load_estimator(offender_path)
    victim = load_estimator(victim_path)
    
    # Extract coefficients
    off_coefs, off_ses = extract_coefficients(offender, crime_type)
    vic_coefs, vic_ses = extract_coefficients(victim, crime_type)
    
    # Get sample sizes
    off_n = offender["results"][crime_type]["num_agents"]
    vic_n = victim["results"][crime_type]["num_agents"]
    
    # Find common variables, preserving order from offender model
    common_vars = [v for v in off_coefs.keys() if v in vic_coefs]
    
    # Compute z-tests
    results = []
    for var in common_vars:
        b_off = off_coefs[var]
        se_off = off_ses[var]
        b_vic = vic_coefs[var]
        se_vic = vic_ses[var]
        
        z, p, sig = compute_ztest(b_off, se_off, b_vic, se_vic)
        
        results.append({
            "variable": var,
            "β_offender": b_off,
            "SE_offender": se_off,
            "β_victim": b_vic,
            "SE_victim": se_vic,
            "diff": b_off - b_vic,
            "z": z,
            "p_value": p,
            "sig": sig
        })
    
    df = pd.DataFrame(results)
    
    if print_results:
        print(f"\n{'='*100}")
        print(f"Comparison: {Path(offender_path).stem} vs {Path(victim_path).stem}")
        print(f"Crime type: {crime_type}")
        print(f"Sample sizes: Offenders N={off_n:,}, Victims N={vic_n:,}")
        print(f"{'='*100}")
        print(f"\n{'Variable':<35} {'β_off':>10} {'β_vic':>10} {'Diff':>10} {'z':>10} {'p-value':>12} {'Sig':>5}")
        print("-" * 100)
        
        for _, row in df.iterrows():
            print(f"{row['variable']:<35} {row['β_offender']:>10.4f} {row['β_victim']:>10.4f} "
                  f"{row['diff']:>10.4f} {row['z']:>10.2f} {row['p_value']:>12.4f} {row['sig']:>5}")
        
        # Summary
        sig_count = (df['p_value'] < 0.05).sum()
        print(f"\n{sig_count} of {len(df)} coefficients significantly different at p < 0.05")
    
    return df


def run_all_comparisons(data_root: str = "data/estimators/biz", crime_type: str = "all_crime_types") -> Dict[str, pd.DataFrame]:
    """
    Run z-test comparisons for all offender-victim model pairs.
    
    Returns:
        Dictionary mapping comparison name to results DataFrame
    """
    data_root = Path(data_root)
    
    # Define comparison pairs: (offender_path, victim_path, comparison_name)
    comparisons = [
        # Biz models - simple business features
        (data_root / "offenders_biz.json",
         data_root / "victims_biz.json",
         "biz"),
        
        # Biz models - simple business with interactions
        (data_root / "offenders_biz_int.json",
         data_root / "victims_biz_int.json",
         "biz_int"),
        
        # Biz models - log employment
        (data_root / "offenders_lnemp.json",
         data_root / "victims_lnemp.json",
         "lnemp"),
        
        # Biz models - log employment with interactions
        (data_root / "offenders_lnemp_int.json",
         data_root / "victims_lnemp_int.json",
         "lnemp_int"),
        
        # Biz models - full business features
        (data_root / "offenders_full_biz.json",
         data_root / "victims_full_biz.json",
         "full_biz"),
        
        # Biz models - full business with interactions
        (data_root / "offenders_full_biz_int.json",
         data_root / "victims_full_biz_int.json",
         "full_biz_int"),
    ]
    
    all_results = {}
    
    for off_path, vic_path, name in comparisons:
        # Check if both files exist
        if not off_path.exists():
            print(f"Skipping {name}: {off_path} not found")
            continue
        if not vic_path.exists():
            print(f"Skipping {name}: {vic_path} not found")
            continue
        
        try:
            df = compare_models(str(off_path), str(vic_path), crime_type=crime_type)
            all_results[name] = df
        except Exception as e:
            print(f"Error comparing {name}: {e}")
    
    return all_results


def export_results_to_csv(
    results: Dict[str, pd.DataFrame],
    output_path: str = "data/tables/ztest_comparisons.csv"
):
    """Export all comparison results to a single CSV file."""
    dfs = []
    for name, df in results.items():
        df_copy = df.copy()
        df_copy.insert(0, "comparison", name)
        dfs.append(df_copy)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\nResults exported to {output_path}")


def export_results_to_latex(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "data/tables"
):
    """Export comparison results to LaTeX tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in results.items():
        # Create formatted LaTeX table
        latex_df = df[["variable", "β_offender", "β_victim", "diff", "z", "sig"]].copy()
        latex_df.columns = ["Variable", r"$\beta_{off}$", r"$\beta_{vic}$", "Diff", "z", ""]
        
        # Format numbers
        for col in [r"$\beta_{off}$", r"$\beta_{vic}$", "Diff"]:
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.3f}")
        latex_df["z"] = latex_df["z"].apply(lambda x: f"{x:.2f}")
        
        latex_str = latex_df.to_latex(
            index=False,
            escape=False,
            column_format="l" + "r" * 5,
            caption=f"Z-test comparison: {name} model (offenders vs victims)",
            label=f"tab:ztest_{name}"
        )
        
        output_path = output_dir / f"ztest_{name}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        print(f"LaTeX table exported to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute z-tests for model comparisons")
    parser.add_argument("--data-root", default="data/estimators/biz", 
                        help="Root directory for estimator files")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--export-latex", action="store_true",
                        help="Export results to LaTeX tables")
    parser.add_argument("--crime-type", default="all_crime_types",
                        help="Crime type to analyze")
    
    args = parser.parse_args()
    
    # Run all comparisons
    results = run_all_comparisons(args.data_root, crime_type=args.crime_type)
    
    # Export if requested
    if args.export_csv:
        export_results_to_csv(results)
    
    if args.export_latex:
        export_results_to_latex(results)
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY OF SIGNIFICANT DIFFERENCES (p < 0.05)")
    print("=" * 100)
    
    for name, df in results.items():
        sig_vars = df[df['p_value'] < 0.05]['variable'].tolist()
        print(f"\n{name}: {len(sig_vars)} significant differences")
        if sig_vars:
            for var in sig_vars:
                row = df[df['variable'] == var].iloc[0]
                direction = "off > vic" if row['diff'] > 0 else "off < vic"
                print(f"  - {var}: z={row['z']:.2f}, {direction}")

