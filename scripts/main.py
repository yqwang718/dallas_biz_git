import json
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from typing import List, Tuple, Union
import argparse
import yaml

from dcm.protocols import (
    AgentFeatures,
    BlockFeatures,
    Estimators,
    load_data,
    Config,
    make_args,
    BlockAggregatedBizFeatures,
    nonzero_features,
)
from dcm.models import (
    dcm_model_chunked_sum,
    dcm_model_interactions_chunked_sum,
    dcm_model_samples,
    dcm_model_interactions_samples,
    dcm_model_network_interactions_chunked_sum,
    dcm_model_network_interactions_samples,
)
from dcm.mle_utils import calculate_se, calculate_bic

# Set up logger
logger = logging.getLogger(__name__)


def extract_race_income_data(
    blocks: List[BlockFeatures],
    agents: List[AgentFeatures],
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Extract race distributions and income data from blocks, and race onehots from agents.

    Returns:
        Tuple of ((block_race_dists, race_agent_onehots), block_incomes) as JAX arrays
    """
    race_dists = []
    incomes = []

    # Get all unique race keys from all blocks and sort them alphabetically
    all_race_keys = set()
    for block in blocks:
        if block.racial_dist:
            all_race_keys.update(block.racial_dist.keys())
    race_order = sorted(all_race_keys)

    if not race_order:
        raise ValueError("No racial distribution data found in any block")

    for i, block in enumerate(blocks):
        # Racial distribution - require it to be present
        if not block.racial_dist:
            raise ValueError(
                f"Block at index {i} (id: {getattr(block, 'block_id', 'unknown')}) "
                f"is missing racial_dist data"
            )

        race_vec = [block.racial_dist.get(race, 0.0) for race in race_order]
        race_dists.append(race_vec)

        # Income - require it to be present
        if block.log_median_income is None:
            raise ValueError(
                f"Block at index {i} (id: {getattr(block, 'block_id', 'unknown')}) "
                f"is missing log_median_income data"
            )
        incomes.append(block.log_median_income)

    # Extract agent race onehots
    race_agent_onehots = []
    for i, agent in enumerate(agents):
        if not agent.race:
            raise ValueError(
                f"Agent at index {i} (id: {getattr(agent, 'agent_id', 'unknown')}) "
                f"is missing race data"
            )

        # Create one-hot vector for agent race
        onehot = [1.0 if race == agent.race else 0.0 for race in race_order]
        race_agent_onehots.append(onehot)

    return (jnp.array(race_dists), jnp.array(race_agent_onehots)), jnp.array(incomes)


def extract_extra_features(
    blocks: Union[List[BlockFeatures], List[BlockAggregatedBizFeatures]],
) -> Tuple[jnp.ndarray, List[str]]:
    if not blocks or not blocks[0].extra_features:
        # No blocks or no extra features in first block
        return jnp.array([]).reshape(len(blocks), 0), []

    # Get feature names from first block
    extra_feature_names = sorted(blocks[0].extra_features.keys())

    # Extract extra features for each block, validating keys match
    extra_features = []
    for i, block in enumerate(blocks):
        if not block.extra_features:
            raise ValueError(
                f"Block at index {i} has no extra_features, but first block has keys: {extra_feature_names}"
            )

        # Check that keys match exactly
        block_keys = set(block.extra_features.keys())
        expected_keys = set(extra_feature_names)
        if block_keys != expected_keys:
            raise ValueError(
                f"Block at index {i} has different extra_features keys. "
                f"Expected: {expected_keys}, Got: {block_keys}"
            )

        # Extract features in sorted order
        feature_vec = [block.extra_features[key] for key in extra_feature_names]
        extra_features.append(feature_vec)

    return jnp.array(extra_features), extra_feature_names


def prepare_data(
    agents: List[AgentFeatures],
    blocks: Union[List[BlockFeatures], List[BlockAggregatedBizFeatures]],
    feature_names: List[str],
    model_type: str = "base",
    include_extra_features: bool = False,
    control_names: List[str] = None,
    network_feature_names: List[str] = None,  # New: explicit network features
) -> Tuple[jnp.ndarray, ...]:
    """Prepare data for DCM optimization using make_args directly."""

    # Extract common block data (used by both model types)
    block_coords = make_args(blocks, ["home_coord"])[0]

    # Handle empty feature_names case
    if feature_names:
        features = make_args(blocks, feature_names, stack=True)[0]
    else:
        # Create empty features array with shape (num_blocks, 0)
        features = jnp.zeros((len(blocks), 0))

    # Extract control features if specified (only for interactions model)
    controls = None
    all_control_names = []
    if model_type == "interactions" and control_names:
        controls = make_args(blocks, control_names, stack=True)[0]
        all_control_names = control_names.copy()
        logger.info(f"Extracted {len(control_names)} control features: {control_names}")

    # Handle network features and extra features
    all_feature_names = feature_names.copy()
    network_features_array = None
    all_network_feature_names = []
    
    # Extract network features if specified (for network_interactions model)
    if model_type == "network_interactions" and network_feature_names:
        # Extract network features from extra_features
        extra_features_dict, extra_feature_names_all = extract_extra_features(blocks)
        
        # Find requested network features
        network_feature_indices = []
        for nf_name in network_feature_names:
            if nf_name in extra_feature_names_all:
                idx = extra_feature_names_all.index(nf_name)
                network_feature_indices.append(idx)
                all_network_feature_names.append(nf_name)
            else:
                logger.warning(f"Network feature '{nf_name}' not found in extra_features")
        
        if network_feature_indices:
            network_features_array = extra_features_dict[:, network_feature_indices]
            logger.info(
                f"Extracted {len(all_network_feature_names)} network features: {all_network_feature_names}"
            )
        else:
            # No network features found, use empty array
            network_features_array = jnp.zeros((len(blocks), 0))
            logger.warning("No network features found, using empty array")
    
    # Handle extra features if requested (for base/interactions models)
    if include_extra_features and model_type != "network_interactions":
        extra_features, extra_feature_names = extract_extra_features(blocks)
        if extra_feature_names:
            # Concatenate extra features to the main features array
            features = jnp.concatenate([features, extra_features], axis=1)
            all_feature_names.extend(extra_feature_names)
            logger.info(
                f"Added {len(extra_feature_names)} extra features: {extra_feature_names}"
            )

    if model_type == "network_interactions":
        # Network interactions model: similar to base but with network features separated
        # Get agent data
        agent_home_ids, agent_coords, chosen_block_ids, chosen_block_coords = make_args(
            agents,
            [
                "home_block_id",
                "home_coord",
                "incident_block_id",
                "incident_block_coord",
            ],
        )
        
        # Extract race and income data
        (block_race_dists, agent_race_onehots), block_incomes = (
            extract_race_income_data(blocks, agents)
        )
        
        # Convert to JAX arrays
        agent_home_idx = jnp.array(agent_home_ids, dtype=jnp.int32)
        agent_coords_jax = jnp.array(agent_coords, dtype=jnp.float32)
        chosen_block_idx = jnp.array(chosen_block_ids, dtype=jnp.int32)
        chosen_block_coords_jax = jnp.array(chosen_block_coords, dtype=jnp.float32)
        block_coords_jax = jnp.array(block_coords, dtype=jnp.float32)
        
        # Control features are the standard block features
        control_features_array = features if features.shape[1] > 0 else jnp.zeros((len(blocks), 0))
        
        return (
            agent_coords_jax,
            block_coords_jax,
            block_race_dists,
            agent_race_onehots,
            block_incomes,
            network_features_array if network_features_array is not None else jnp.zeros((len(blocks), 0)),
            control_features_array,
            agent_home_idx,
            chosen_block_idx,
            chosen_block_coords_jax,
            all_network_feature_names,
            all_feature_names,
        )
    
    elif model_type == "interactions":
        # Get agent data for interactions model (no home_block_id needed)
        agent_coords, chosen_block_ids = make_args(
            agents, ["home_coord", "incident_block_id"]
        )

        # Convert to JAX arrays
        agent_coords_jax = jnp.array(agent_coords, dtype=jnp.float32)
        chosen_block_idx = jnp.array(chosen_block_ids, dtype=jnp.int32)
        block_coords_jax = jnp.array(block_coords, dtype=jnp.float32)

        return (
            agent_coords_jax,
            chosen_block_idx,
            block_coords_jax,
            controls if controls is not None else jnp.zeros((len(blocks), 0)),
            features,
            all_control_names,
            all_feature_names,
        )

    else:  # base model
        # Get agent data
        agent_home_ids, agent_coords, chosen_block_ids, chosen_block_coords = make_args(
            agents,
            [
                "home_block_id",
                "home_coord",
                "incident_block_id",
                "incident_block_coord",
            ],
        )

        # Extract race and income data
        (block_race_dists, agent_race_onehots), block_incomes = (
            extract_race_income_data(blocks, agents)
        )

        # Convert to JAX arrays
        agent_home_idx = jnp.array(agent_home_ids, dtype=jnp.int32)
        agent_coords_jax = jnp.array(agent_coords, dtype=jnp.float32)
        chosen_block_idx = jnp.array(chosen_block_ids, dtype=jnp.int32)
        chosen_block_coords_jax = jnp.array(chosen_block_coords, dtype=jnp.float32)
        block_coords_jax = jnp.array(block_coords, dtype=jnp.float32)

        return (
            agent_coords_jax,
            block_coords_jax,
            block_race_dists,
            agent_race_onehots,
            block_incomes,
            features,
            agent_home_idx,
            chosen_block_idx,
            chosen_block_coords_jax,
            all_feature_names,
        )


def to_estimators(
    params: jnp.ndarray,
    control_names: List[str],
    feature_names: List[str],
    model_type: str = "base",
    interaction: bool = True,
    network_feature_names: List[str] = None,
) -> Estimators:
    """Convert parameter vector back to Estimators object."""
    if model_type == "network_interactions":
        # Network interactions model
        beta_distance = float(params[0])
        beta_race = float(params[1])
        beta_income = float(params[2])
        
        num_network = len(network_feature_names) if network_feature_names else 0
        num_controls = len(control_names)
        
        # Extract network feature betas (main effects)
        betas_network = params[3:3+num_network]
        
        # Extract interaction betas if interaction is True
        if interaction:
            betas_interactions = params[3+num_network:3+2*num_network]
            betas_controls = params[3+2*num_network:]
        else:
            betas_controls = params[3+num_network:]
        
        # Create features dict
        features = {}
        
        # Add network features (main effects)
        for idx, name in enumerate(network_feature_names):
            features[name] = float(betas_network[idx])
        
        # Add interaction terms if enabled
        if interaction:
            for idx, name in enumerate(network_feature_names):
                features[f"interaction_distance_x_{name}"] = float(betas_interactions[idx])
        
        # Add control features
        for idx, name in enumerate(control_names):
            features[name] = float(betas_controls[idx])
    
    elif model_type == "interactions":
        beta_distance = float(params[0])
        beta_race = None
        beta_income = None

        num_controls = len(control_names)
        num_features = len(feature_names)

        # Extract control betas
        betas_controls = params[1 : 1 + num_controls] if num_controls > 0 else []

        # Extract feature betas
        features_start = 1 + num_controls
        betas_features = params[features_start : features_start + num_features]

        # Create features dict (include controls in the features dict)
        features = {}

        # Add control coefficients
        for idx, name in enumerate(control_names):
            features[name] = float(betas_controls[idx])

        # Add feature coefficients
        for idx, name in enumerate(feature_names):
            features[name] = float(betas_features[idx])

        # Add interactions only if interaction is True
        if interaction:
            betas_interactions = params[features_start + num_features :]
            features.update(
                {
                    f"interaction_{name}": float(betas_interactions[idx])
                    for idx, name in enumerate(feature_names)
                }
            )
    else:  # base model
        beta_distance = float(params[0])
        beta_race = float(params[1])
        beta_income = float(params[2])
        betas_features = params[3:]

        # Create features dict using the same approach
        features = {
            name: float(betas_features[idx]) for idx, name in enumerate(feature_names)
        }

    return Estimators(
        distance=beta_distance,
        race=beta_race,
        income=beta_income,
        features=features if features else None,
    )


def optimize_dcm_model(
    agent_features: List[AgentFeatures],
    block_features: Union[List[BlockFeatures], List[BlockAggregatedBizFeatures]],
    config: Config,
) -> Tuple[Estimators, Estimators, float, bool, float]:
    """Run DCM model optimization and compute standard errors and BIC.

    Returns:
        Tuple of (estimators, standard_errors, final_loss, converged, bic)
    """

    # Extract what we need from config
    model_type = config.model.model_type
    feature_names = config.model.feature_names
    control_names = (
        config.model.control_names if config.model.model_type in ["interactions", "network_interactions"] else []
    )
    network_feature_names = (
        config.model.network_feature_names if model_type == "network_interactions" else None
    )
    chunk_size = config.optimizer.chunk_size
    max_iter = config.optimizer.max_iter
    distance_interaction = config.model.distance_interaction
    race_interaction = config.model.race_interaction
    income_interaction = config.model.income_interaction
    distance_features_interaction = config.model.distance_features_interaction
    interaction = config.model.interaction if model_type in ["interactions", "network_interactions"] else None
    network_interaction = config.model.network_interaction if model_type == "network_interactions" else None
    include_extra_features = config.model.include_extra_features

    # Prepare data based on model type
    data = prepare_data(
        agent_features,
        block_features,
        feature_names,
        model_type,
        include_extra_features,
        control_names,
        network_feature_names,
    )

    if model_type == "network_interactions":
        (
            agent_coords,
            block_coords,
            block_race_dists,
            agent_race_onehots,
            block_incomes,
            network_features,
            control_features,
            agent_home_ids,
            chosen_block_ids,
            chosen_block_coords,
            returned_network_feature_names,
            returned_control_names,
        ) = data
    elif model_type == "interactions":
        (
            agent_coords,
            chosen_block_ids,
            block_coords,
            controls,
            features,
            returned_control_names,
            returned_feature_names,
        ) = data
    else:  # base model
        (
            agent_coords,
            block_coords,
            block_race_dists,
            agent_race_onehots,
            block_incomes,
            features,
            agent_home_ids,
            chosen_block_ids,
            chosen_block_coords,
            returned_feature_names,
        ) = data

    # Random initialization
    key = jax.random.PRNGKey(42)
    
    if model_type == "network_interactions":
        # For network_interactions: beta_distance + beta_race + beta_income + 
        #                           betas_network + (betas_interactions if network_interaction=True) + betas_controls
        num_network = network_features.shape[1]
        num_controls = control_features.shape[1]
        if network_interaction:
            # 3 core + N_network main + N_network interactions + M_controls
            initial_params = jax.random.normal(key, (3 + 2 * num_network + num_controls,)) * 0.1
        else:
            # 3 core + N_network main + M_controls
            initial_params = jax.random.normal(key, (3 + num_network + num_controls,)) * 0.1
    elif model_type == "interactions":
        num_controls = len(returned_control_names)
        num_features = features.shape[1]
        # For interactions: beta_distance + betas_controls + betas_features + (betas_interactions if interaction=True)
        if interaction:
            initial_params = (
                jax.random.normal(key, (1 + num_controls + 2 * num_features,)) * 0.1
            )
        else:
            initial_params = (
                jax.random.normal(key, (1 + num_controls + num_features,)) * 0.1
            )
    else:
        # For base: beta_distance + beta_race + beta_income + betas_features
        num_features = features.shape[1]
        initial_params = jax.random.normal(key, (3 + num_features,)) * 0.1

    def objective(params):
        """Objective function for optimization."""
        if model_type == "network_interactions":
            model_inputs = (
                params,
                agent_home_ids,
                agent_coords,
                agent_race_onehots,
                chosen_block_ids,
                chosen_block_coords,
                block_coords,
                block_race_dists,
                block_incomes,
                network_features,
                control_features,
            )
            
            return (
                dcm_model_network_interactions_chunked_sum(
                    *model_inputs,
                    chunk_size=chunk_size,
                    distance_interaction=distance_interaction,
                    race_interaction=race_interaction,
                    income_interaction=income_interaction,
                    network_interaction=network_interaction,
                )
                / agent_coords.shape[0]
            )
        
        elif model_type == "interactions":
            model_inputs = (
                params,
                agent_coords,
                chosen_block_ids,
                block_coords,
                controls,
                features,
            )

            return (
                dcm_model_interactions_chunked_sum(
                    *model_inputs,
                    chunk_size=chunk_size,
                    distance_interaction=distance_interaction,
                    distance_features_interaction=distance_features_interaction,
                    interaction=interaction,
                )
                / agent_coords.shape[0]
            )

        else:  # base model
            model_inputs = (
                params,
                agent_home_ids,
                agent_coords,
                agent_race_onehots,
                chosen_block_ids,
                chosen_block_coords,
                block_coords,
                block_race_dists,
                block_incomes,
                features,
            )

            return (
                dcm_model_chunked_sum(
                    *model_inputs,
                    chunk_size=chunk_size,
                    distance_interaction=distance_interaction,
                    race_interaction=race_interaction,
                    income_interaction=income_interaction,
                )
                / agent_coords.shape[0]
            )

    # Run optimization
    logger.info("Running optimization...")
    result = minimize(
        objective, initial_params, method="BFGS", options={"maxiter": max_iter}
    )

    # Prepare arguments for SE and BIC calculation
    from functools import partial
    
    if model_type == "network_interactions":
        model_fn_partial = partial(
            dcm_model_network_interactions_samples,
            distance_interaction=distance_interaction,
            race_interaction=race_interaction,
            income_interaction=income_interaction,
            network_interaction=network_interaction,
        )
        se_args = (
            agent_home_ids,
            agent_coords,
            agent_race_onehots,
            chosen_block_ids,
            chosen_block_coords,
            block_coords,
            block_race_dists,
            block_incomes,
            network_features,
            control_features,
        )
    elif model_type == "interactions":
        model_fn_partial = partial(
            dcm_model_interactions_samples,
            distance_interaction=distance_interaction,
            distance_features_interaction=distance_features_interaction,
            interaction=interaction,
        )
        se_args = (agent_coords, chosen_block_ids, block_coords, controls, features)
    else:  # base model
        model_fn_partial = partial(
            dcm_model_samples,
            distance_interaction=distance_interaction,
            race_interaction=race_interaction,
            income_interaction=income_interaction,
        )
        se_args = (
            agent_home_ids,
            agent_coords,
            agent_race_onehots,
            chosen_block_ids,
            chosen_block_coords,
            block_coords,
            block_race_dists,
            block_incomes,
            features,
        )

    # Calculate standard errors
    logger.info("Computing standard errors...")
    se = calculate_se(model_fn_partial, result.x, se_args, chunk_size)

    # Calculate BIC
    logger.info("Computing BIC...")
    bic = calculate_bic(model_fn_partial, result.x, se_args, chunk_size)

    # Convert results to Estimators objects
    if model_type == "network_interactions":
        estimators = to_estimators(
            result.x,
            returned_control_names,
            [],  # No feature_names for network_interactions
            model_type=model_type,
            interaction=network_interaction,
            network_feature_names=returned_network_feature_names,
        )
        standard_errors = to_estimators(
            se,
            returned_control_names,
            [],
            model_type=model_type,
            interaction=network_interaction,
            network_feature_names=returned_network_feature_names,
        )
    else:
        returned_control_names_for_to_est = returned_control_names if model_type == "interactions" else []
        estimators = to_estimators(
            result.x,
            returned_control_names_for_to_est,
            returned_feature_names,
            model_type=model_type,
            interaction=interaction,
        )
        standard_errors = to_estimators(
            se,
            returned_control_names_for_to_est,
            returned_feature_names,
            model_type=model_type,
            interaction=interaction,
        )

    return estimators, standard_errors, float(result.fun), result.success, bic


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file and create Config object."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert YAML dict to Config object
    return Config(**config_dict)


def main():
    """
    Example usage of the DCM optimization routine.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DCM Model Optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Define part one crime types (from 2_feature_extraction.ipynb)
    crime_types = [
        "burglary_breaking_entering",
        "motor_vehicle_theft",
        "larceny_theft_offenses",
        "assault_offenses",
        "robbery",
        "drug_narcotic_violations",
    ]

    # Subset excluding burglary and drug crimes
    crime_types_subset = [
        "motor_vehicle_theft",
        "larceny_theft_offenses",
        "assault_offenses",
        "robbery",
    ]

    # For individual analyses: use subset for victims, all for offenders
    individual_crime_types = (
        crime_types_subset if config.data.agent.startswith("victims") else crime_types
    )

    # Create list of analyses to run: individual crime types + all combined
    analyses = [(crime_type, crime_type) for crime_type in individual_crime_types]
    # For "all_crime_types", always use the subset (4 types)
    analyses.append(("all_crime_types", crime_types_subset))

    # Dictionary to store all results
    all_results = {}

    # Load file paths once
    agent_file = f"{config.data.data_root}/{config.data.agent}.jsonl"
    block_file = f"{config.data.data_root}/{config.data.block}.jsonl"

    # Run analyses
    for label, crime_filter in analyses:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running analysis for: {label}")
        logger.info(f"{'='*60}")

        # Update filter dict for this analysis
        agent_filter_dict = config.data.agent_filter_dict or {}
        agent_filter_dict["crime_type"] = crime_filter

        # Load agents
        agents = load_data(agent_file, AgentFeatures, agent_filter_dict)
        logger.info(f"Loaded {len(agents)} {config.data.agent} for {label}")

        # Skip if no agents found
        if len(agents) == 0:
            logger.warning(f"No agents found for {label}, skipping...")
            continue

        # Load blocks
        logger.info("Loading block features...")
        if config.model.model_type == "interactions":
            blocks = load_data(
                block_file, BlockAggregatedBizFeatures, config.data.block_filter_dict
            )
        else:
            blocks = load_data(block_file, BlockFeatures, config.data.block_filter_dict)
        logger.info(f"Loaded {len(blocks)} blocks")

        # Apply nonzero features filtering if configured
        if config.data.filter_nonzero_features:
            original_count = len(agents)
            agents = nonzero_features(agents, blocks, config.model.feature_names)
            logger.info(
                f"Filtered agents based on non-zero features: {original_count} -> {len(agents)} "
                f"({len(agents)/original_count*100:.1f}% retained)"
            )

            # Skip if no agents remain after filtering
            if len(agents) == 0:
                logger.warning(
                    f"No agents remain after non-zero feature filtering for {label}, skipping..."
                )
                continue

        # Run optimization
        try:
            estimators, standard_errors, final_loss, converged, bic = (
                optimize_dcm_model(
                    agents,
                    blocks,
                    config,
                )
            )

            # Store results
            result_entry = {
                "estimators": estimators.model_dump(),
                "standard_errors": standard_errors.model_dump(),
                "final_loss": float(final_loss),
                "converged": bool(converged),
                "bic": float(bic),
                "num_agents": len(agents),
            }

            # Add crime types info for the combined analysis
            if label == "all_crime_types":
                result_entry["crime_types_included"] = crime_types_subset

            all_results[label] = result_entry
            logger.info(f"Completed analysis for {label}")

        except Exception as e:
            logger.error(f"Error processing {label}: {str(e)}")
            all_results[label] = {"error": str(e)}

    # Add metadata to results
    final_results = {
        "metadata": {
            "model_type": config.model.model_type,
            "distance_interaction": config.model.distance_interaction,
            "race_interaction": (
                config.model.race_interaction
                if config.model.model_type == "base"
                else None
            ),
            "income_interaction": (
                config.model.income_interaction
                if config.model.model_type == "base"
                else None
            ),
            "distance_features_interaction": (
                config.model.distance_features_interaction
                if config.model.model_type == "interactions"
                else None
            ),
            "interaction": (
                config.model.interaction
                if config.model.model_type == "interactions"
                else None
            ),
            "config_used": config.model_dump(),
            "individual_crime_types_analyzed": individual_crime_types,
            "all_crime_types_pooled": crime_types_subset,
        },
        "results": all_results,
    }

    # Save results
    if config.output_file:
        with open(config.output_file, "w") as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"\nAll results saved to {config.output_file}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY OF RESULTS")
        logger.info("=" * 60)
        for label, result in all_results.items():
            if "error" not in result:
                logger.info(
                    f"{label}: "
                    f"n={result.get('num_agents', 'N/A')}, "
                    f"converged={result.get('converged', 'N/A')}, "
                    f"BIC={result.get('bic', 'N/A'):.2f}"
                )
            else:
                logger.info(f"{label}: ERROR - {result['error']}")
    else:
        logger.info("No output file specified - results not saved")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
