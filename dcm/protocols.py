import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Union, Type, TypeVar, List

# import jax.numpy as jnp  # Temporarily commented out for summarizer
import numpy as np


T = TypeVar("T", bound=BaseModel)


class AgentFeatures(BaseModel):
    agent_id: Optional[int] = None
    home_block_id: Optional[int] = None
    home_coord: Optional[tuple[float, ...]] = None
    race: Optional[str] = None
    crime_type: Optional[str] = None
    incident_block_id: Optional[int] = None
    incident_block_coord: Optional[tuple[float, ...]] = None


class BlockFeatures(BaseModel):
    block_id: Optional[int] = None
    home_coord: Optional[tuple[float, ...]] = None
    racial_dist: Optional[dict[str, float]] = None
    log_median_income: Optional[float] = None
    log_total_population: Optional[float] = None
    log_total_employees: Optional[float] = None
    log_landsize: Optional[float] = None
    avg_household_size: Optional[float] = None
    home_owners_perc: Optional[float] = None
    underage_perc: Optional[float] = None
    log_attractions: Optional[float] = None
    log_transit_stops: Optional[float] = None
    extra_features: Optional[dict[str, float]] = None


class BlockAggregatedBizFeatures(BaseModel):
    block_id: Optional[int] = None
    home_coord: Optional[tuple[float, ...]] = None
    log_consumer_control: Optional[float] = None
    log_white_collar_control: Optional[float] = None
    log_blue_collar_control: Optional[float] = None
    log_consumer: Optional[float] = None
    log_white_collar: Optional[float] = None
    log_blue_collar: Optional[float] = None
    consumer_hetero: Optional[float] = None
    log_emp_consumer: Optional[float] = None
    log_emp_white_collar: Optional[float] = None
    log_emp_blue_collar: Optional[float] = None
    log_emp_consumer_hetero: Optional[float] = None
    extra_features: Optional[dict[str, float]] = None


class Estimators(BaseModel):
    distance: Optional[float] = None
    race: Optional[float] = None
    income: Optional[float] = None
    features: Optional[dict[str, float]] = None  # Dynamic features dict


def load_data(
    file_path: Union[str, Path],
    model_class: Type[T],
    filter_dict: Optional[dict] = None,
    filter_func_dict: Optional[dict] = None,
) -> list[T]:
    """
    Load data from a JSONL file and create instances of the specified pydantic model.

    Args:
        file_path: Path to the JSONL file
        model_class: The pydantic model class to instantiate (e.g., AgentFeatures, BlockFeatures)
        filter_dict: Optional dictionary for filtering entries based on field values.
                    Values can be single items for exact match, or lists for membership check.
                    Example: {'race': 'WHITE', 'crime_type': ['drug', 'theft', 'assault']}
        filter_func_dict: Optional dictionary for filtering entries using functions.
                         Each function is applied to the field value from the pydantic instance.
                         Example: {'agent_id': lambda x: x > 100, 'log_median_income': lambda x: x is not None and x > 5.0}

    Returns:
        List of model instances that match the filter criteria
    """
    data = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            json_data = json.loads(line)

            # Apply filter if provided
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    data_value = json_data.get(key)
                    if isinstance(value, list):
                        # If filter value is a list, check if data value is in the list
                        if data_value not in value:
                            match = False
                            break
                    else:
                        # Otherwise, do exact equality check
                        if data_value != value:
                            match = False
                            break
                if not match:
                    continue

            # Create model instance (pydantic will handle partial matches gracefully)
            instance = model_class(**json_data)

            # Apply function-based filter if provided
            if filter_func_dict:
                if not all(
                    func(getattr(instance, key))
                    for key, func in filter_func_dict.items()
                ):
                    continue

            data.append(instance)

    return data


def make_args(
    instances: list[T],
    field_names: Union[list[str], list[Union[str, list[str]]]],
    stack: bool = False,
) -> tuple[np.ndarray, ...]:
    if not instances:
        # Return empty arrays based on field structure
        if stack:
            return (np.array([]),)
        else:
            num_outputs = len(
                [f for f in field_names if not isinstance(f, list)]
            ) + len([f for f in field_names if isinstance(f, list)])
            return tuple(np.array([]) for _ in range(num_outputs))

    # Process field names to handle grouped fields
    field_groups = []
    if stack:
        # If stack=True, treat all fields as one group
        flat_fields = []
        for f in field_names:
            if isinstance(f, list):
                flat_fields.extend(f)
            else:
                flat_fields.append(f)
        field_groups = [flat_fields]
    else:
        # Otherwise, process each field/group separately
        for f in field_names:
            if isinstance(f, list):
                field_groups.append(f)
            else:
                field_groups.append([f])

    arrays = []

    for group in field_groups:
        group_arrays = []

        for field_name in group:
            # Extract values for this field from all instances
            values = []

            for instance in instances:
                value = getattr(instance, field_name, None)

                # Handle None values
                if value is None:
                    # For coordinates, use NaN; for IDs, use -1
                    if "coord" in field_name:
                        value = (np.nan, np.nan)  # Assuming 2D coordinates
                    elif "id" in field_name:
                        value = -1
                    else:
                        value = np.nan

                # Convert tuples to lists for consistent array creation
                if isinstance(value, tuple):
                    value = list(value)

                values.append(value)

            # Convert to JAX array
            try:
                array = np.array(values)
            except Exception as e:
                array = np.array(values, dtype=object)

            # Ensure array is at least 2D for stacking
            if array.ndim == 1:
                array = array[:, None]

            group_arrays.append(array)

        # Stack arrays in this group if multiple fields
        if len(group_arrays) > 1:
            # Stack along last dimension
            stacked = np.concatenate(group_arrays, axis=-1)
            arrays.append(stacked)
        else:
            # Single field, just append (remove extra dimension if it was added)
            arr = group_arrays[0]
            if arr.shape[-1] == 1 and len(group) == 1:
                arr = arr.squeeze(-1)
            arrays.append(arr)

    return tuple(arrays)


def nonzero_features(
    agents: List[AgentFeatures],
    blocks: Union[List[BlockFeatures], List[BlockAggregatedBizFeatures]],
    feature_names: List[str],
) -> List[AgentFeatures]:
    """
    Filter agent features to keep only those whose incident_block_id links to blocks
    with at least one of the specified features being non-zero.

    Args:
        agents: List of AgentFeatures to filter
        blocks: List of BlockFeatures or BlockAggregatedBizFeatures
        feature_names: List of feature field names to check for non-zero values

    Returns:
        Filtered list of AgentFeatures
    """
    # Create mapping from block_id to block object for efficient lookup
    block_map = {
        block.block_id: block for block in blocks if block.block_id is not None
    }

    filtered_agents = []

    for agent in agents:
        # Skip if agent has no incident_block_id
        if agent.incident_block_id is None:
            continue

        # Check if the incident block exists in our blocks
        if agent.incident_block_id not in block_map:
            raise ValueError(f"{agent.incident_block_id=} not in block id set")

        block = block_map[agent.incident_block_id]

        # Check if at least one specified feature is non-zero
        has_nonzero = False
        for feature_name in feature_names:
            # Get the feature value from the block
            feature_value = getattr(block, feature_name, None)

            # Check if feature is non-zero (not None, not NaN, not zero)
            if feature_value is not None:
                if isinstance(feature_value, (int, float)):
                    if not np.isnan(feature_value) and feature_value != 0:
                        has_nonzero = True
                        break

        # Keep agent only if at least one feature is non-zero
        if has_nonzero:
            filtered_agents.append(agent)

    return filtered_agents


class DataConfig(BaseModel):
    data_root: str = "data/features"
    agent: str = "offenders"  # "offenders" or "victims"
    block: str = "block"
    agent_filter_dict: Optional[dict] = None
    block_filter_dict: Optional[dict] = None
    filter_nonzero_features: bool = (
        False  # Whether to filter agents based on non-zero block features
    )


class ModelConfig(BaseModel):
    model_type: str = "base"  # "base", "interactions", or "network_interactions"
    distance_interaction: str = "l2_log"
    race_interaction: Optional[str] = "dissimilarity"
    income_interaction: Optional[str] = "abs_diff"
    distance_features_interaction: str = "l2_log_product"  # For interactions model
    interaction: bool = (
        True  # Whether to include interaction terms in interactions model
    )
    feature_names: list[str] = [
        "log_total_population",
        "avg_household_size",
        "log_total_employees",
        "log_attractions",
        "log_transit_stops",
    ]  # Features to use in the model
    control_names: list[str] = (
        []
    )  # Control features (not interacted with distance) for interactions model
    include_extra_features: bool = False
    
    # Network-specific fields
    network_feature_names: Optional[list[str]] = None  # Network features for network_interactions model
    network_interaction: bool = False  # Whether to include distance × network interactions


class OptimizerConfig(BaseModel):
    chunk_size: int = 16384
    max_iter: int = 1000


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    output_file: Optional[str] = None


if __name__ == "__main__":
    print(
        load_data(
            "../data/features/offenders.jsonl",
            AgentFeatures,
            {"race": "ASIAN", "crime_type": "others"},
        )
    )
