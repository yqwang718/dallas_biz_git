#!/usr/bin/env python3
"""
Convert JSON estimator files to CSV or LaTeX format with significance stars.

Usage:
    python saver.py input_dir output_dir [--latex]
"""

import json
import sys
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
from table_utils import (
    calculate_significance_stars,
    format_coefficient,
    dataframe_to_latex,
)


def extract_estimators_from_json(
    json_data: Dict,
) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, float], Dict[str, int]]:
    results = json_data.get("results", {})

    # Collect all unique estimator names in order of appearance
    estimator_order = []
    estimator_set = set()
    formatted_data = {}
    bic_data = {}
    num_agents_data = {}

    # Track which estimators have at least one non-null value
    non_null_estimators = set()

    for crime_type, crime_data in results.items():
        if not isinstance(crime_data, dict) or "estimators" not in crime_data:
            continue

        estimators = crime_data["estimators"]
        std_errors = crime_data["standard_errors"]

        formatted_data[crime_type] = {}

        # Extract BIC and num_agents
        if "bic" in crime_data:
            bic_data[crime_type] = crime_data["bic"]
        if "num_agents" in crime_data:
            num_agents_data[crime_type] = crime_data["num_agents"]

        # Process non-feature estimators in a specific order
        for key in ["distance", "race", "income"]:
            if key in estimators and estimators[key] is not None:
                if key not in estimator_set:
                    estimator_order.append(key)
                    estimator_set.add(key)
                non_null_estimators.add(key)
                formatted_data[crime_type][key] = format_coefficient(
                    estimators[key], std_errors.get(key)
                )

        # Process feature estimators
        if "features" in estimators and isinstance(estimators["features"], dict):
            for feature_name, feature_value in estimators["features"].items():
                if feature_value is not None:
                    if feature_name not in estimator_set:
                        estimator_order.append(feature_name)
                        estimator_set.add(feature_name)
                    non_null_estimators.add(feature_name)
                    feature_se = std_errors.get("features", {}).get(feature_name)
                    formatted_data[crime_type][feature_name] = format_coefficient(
                        feature_value, feature_se
                    )

    # Only keep estimators that have at least one non-null value, maintaining order
    estimator_names = [name for name in estimator_order if name in non_null_estimators]

    return estimator_names, formatted_data, bic_data, num_agents_data


def create_dataframe(
    estimator_names: List[str],
    formatted_data: Dict[str, Dict[str, str]],
    bic_data: Dict[str, float],
    num_agents_data: Dict[str, int],
) -> pd.DataFrame:
    """Create a pandas DataFrame from the formatted data, including BIC and num_agents."""
    # Create crime type column names
    crime_types = list(formatted_data.keys())

    # Build the dataframe for estimators
    data_dict = {}
    for crime_type in crime_types:
        column_values = []
        for estimator in estimator_names:
            value = formatted_data[crime_type].get(estimator, "")
            column_values.append(value)
        data_dict[crime_type] = column_values

    df = pd.DataFrame(data_dict, index=estimator_names)

    # Add separator row (empty row)
    separator_row = pd.Series([""] * len(crime_types), index=crime_types, name="")

    # Add BIC row
    bic_row_data = []
    for crime_type in crime_types:
        if crime_type in bic_data:
            bic_row_data.append(f"{bic_data[crime_type]:.2f}")
        else:
            bic_row_data.append("")
    bic_row = pd.Series(bic_row_data, index=crime_types, name="BIC")

    # Add num_agents row
    num_agents_row_data = []
    for crime_type in crime_types:
        if crime_type in num_agents_data:
            num_agents_row_data.append(f"{num_agents_data[crime_type]:,}")
        else:
            num_agents_row_data.append("")
    num_agents_row = pd.Series(num_agents_row_data, index=crime_types, name="N")

    # Concatenate all parts
    df = pd.concat(
        [
            df,
            separator_row.to_frame().T,
            bic_row.to_frame().T,
            num_agents_row.to_frame().T,
        ]
    )

    return df


def process_json_file(
    input_file: Path, output_dir: Path, is_latex: bool, caption: str, label_prefix: str
) -> None:
    """Process a single JSON file and save the output."""
    # Load JSON data
    with open(input_file, "r") as f:
        json_data = json.load(f)

    # Extract and format data
    estimator_names, formatted_data, bic_data, num_agents_data = (
        extract_estimators_from_json(json_data)
    )

    if not estimator_names:
        print(f"Warning: No valid estimators found in {input_file}")
        return

    df = create_dataframe(estimator_names, formatted_data, bic_data, num_agents_data)

    # Determine output filename
    base_name = input_file.stem
    output_extension = ".tex" if is_latex else ".csv"
    output_file = output_dir / (base_name + output_extension)

    # Save output
    if is_latex:
        # Use the base name for table label
        label = f"{label_prefix}:{base_name}"
        caption_text = caption.replace("{filename}", base_name)
        # Use dataframe_to_latex with add_midrule_before_row for separator
        latex_output = dataframe_to_latex(
            df,
            caption_text,
            label,
            adjustbox=True,
            add_midrule_before_row="",  # Add midrule before empty row
        )
        with open(output_file, "w") as f:
            f.write(latex_output)
        print(f"LaTeX table saved to {output_file}")
    else:
        df.to_csv(output_file)
        print(f"CSV file saved to {output_file}")

    # Print summary (adjusted to exclude BIC and N rows from estimator count)
    num_estimators = len(df) - 3  # Subtract separator row, BIC row, and N row
    print(
        f"  - Table dimensions: {num_estimators} estimators + BIC/N × {len(df.columns)} crime types"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert all JSON estimator files in a directory to CSV or LaTeX format"
    )
    parser.add_argument("input_dir", help="Input directory containing JSON files")
    parser.add_argument("output_dir", help="Output directory for CSV/LaTeX files")
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output as LaTeX tables (default: CSV)",
    )
    parser.add_argument(
        "--caption",
        default="Estimation Results for {filename}",
        help="Caption for LaTeX tables (use {filename} as placeholder)",
    )
    parser.add_argument(
        "--label-prefix",
        default="tab",
        help="Label prefix for LaTeX tables (will be formatted as prefix:filename)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files in the input directory
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        sys.exit(0)

    print(f"Found {len(json_files)} JSON file(s) in '{input_dir}'")
    print(f"Output format: {'LaTeX' if args.latex else 'CSV'}")
    print(f"Output directory: '{output_dir}'")
    print()

    # Process each JSON file
    for json_file in sorted(json_files):
        print(f"Processing {json_file.name}...")
        try:
            process_json_file(
                json_file, output_dir, args.latex, args.caption, args.label_prefix
            )
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
        print()

    print("Processing complete!")


if __name__ == "__main__":
    main()
