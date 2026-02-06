import pandas as pd


def aggregate_by_prefix(df, col_name, group_name, prefix=None):
    """
    Aggregate a DataFrame by the prefix of a column and return a grouped and filtered result.

    Parameters:
    - df: pd.DataFrame
        Input DataFrame to be aggregated.
    - col_name: str
        Name of the column to use for prefix grouping (e.g., 'blkidfp10').
    - group_name: str
        Name for the new column that stores the prefix grouping (e.g., 'FIPS').
    - prefix: str, optional
        If provided, filters the grouped data to only rows where `col_name` starts with this prefix.

    Returns:
    - pd.DataFrame
        Grouped and optionally filtered DataFrame.
    """
    # Group by prefix (first 12 characters)
    grouped = df.groupby(df[col_name].str[:12]).sum(numeric_only=True)

    # Assign group_name as a column
    grouped[group_name] = grouped.index.values
    grouped = grouped.reset_index(drop=True)

    # Optional filtering by prefix
    if prefix is not None:
        grouped = grouped[df[col_name].str.startswith(prefix)]

    return grouped


def print_stata_variable_labels(file_path):
    """
    Print variable labels (column annotations) from a Stata file if they exist.

    Parameters:
    -----------
    file_path : str
        Path to the Stata file
    """
    try:
        reader = pd.read_stata(file_path, iterator=True)
        var_labels = reader.variable_labels()
        reader.close()

        if var_labels:
            print(f"Variable labels for {file_path}:")
            max_var_len = max(len(var) for var in var_labels.keys())
            for var, label in var_labels.items():
                print(f"  {var:<{max_var_len}} : {label}")
        else:
            print(f"No variable labels found in {file_path}")

    except Exception as e:
        print(f"Error reading variable labels from {file_path}: {e}")
