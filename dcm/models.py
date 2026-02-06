from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, lax

from dcm.interactions import *
import logging

logger = logging.getLogger(__name__)


# TODO: features not applying unary functions yet
@partial(
    jax.jit,
    static_argnames=["distance_interaction", "race_interaction", "income_interaction"],
)
def dcm_model(
    # All betas in one array
    betas,  # (3 + F)
    # Agent related
    agent_home_id,  # ()
    agent_coord,  # (2)
    agent_race_onehot,  # (R)
    chosen_block_id,  # ()
    chosen_block_coord,  # (2) - NEW PARAMETER
    # Block related
    block_coords,  # (C, 2)
    block_race_dists,  # (C, R)
    block_incomes,  # (C)
    # Features
    features,  # (C, F)
    # kwargs
    distance_interaction: str = "l2_log",
    race_interaction: str = "dissimilarity",
    income_interaction: str = "abs_diff",
):
    # Unpack betas
    beta_distance = betas[0]
    beta_race = betas[1]
    beta_income = betas[2]
    betas_features = betas[3:]

    # Use conditional distance: if choosing home block, use agent-agent distance
    # otherwise use the standard agent-block distance
    distances = agent_block_interaction(distance_interaction)(
        agent_coord, block_coords
    )  # (C,)
    actual_distance = agent_agent_interaction(distance_interaction)(
        agent_coord, chosen_block_coord
    )  # ()
    condition = (agent_home_id == chosen_block_id) & (
        jnp.arange(block_coords.shape[0]) == agent_home_id
    )  # (C,)
    distance = jnp.where(condition, actual_distance, distances)  # (C,)

    # Conditional race interaction based on interaction type
    if race_interaction == "threshold":
        race_diss = agent_block_interaction(race_interaction)(
            agent_race_onehot, block_race_dists
        )
    else:
        race_diss = block_block_interaction(race_interaction)(
            agent_home_id, block_race_dists
        )

    income_diss = block_block_interaction(income_interaction)(
        agent_home_id, block_incomes
    )

    logit = (
        beta_distance * distance
        + beta_race * race_diss
        + beta_income * income_diss
        + (betas_features * features).sum(-1)
    )
    logsoftmax = jax.nn.log_softmax(logit)
    return -logsoftmax[chosen_block_id]


@partial(
    jax.jit,
    static_argnames=[
        "distance_interaction",
        "distance_features_interaction",
        "interaction",
    ],
)
def dcm_model_interactions(
    # All betas in one array
    betas,  # (1 + num_controls + 2*F) if interaction=True, (1 + num_controls + F) if interaction=False
    # Agent related
    agent_coord,  # (2)
    chosen_block_id,  # ()
    # Block related
    block_coords,  # (C, 2)
    # Controls (not interacted with distance)
    controls,  # (C, num_controls)
    # Features
    features,  # (C, F)
    # kwargs
    distance_interaction: str = "l2_log",
    distance_features_interaction: str = "l2_log_product",
    interaction: bool = True,
):
    # Unpack betas
    beta_distance = betas[0]
    num_controls = controls.shape[1] if controls.shape[1] > 0 else 0
    num_features = features.shape[1]

    # Extract control betas
    betas_controls = betas[1 : 1 + num_controls] if num_controls > 0 else jnp.array([])

    # Extract feature and interaction betas
    features_start = 1 + num_controls
    if interaction:
        betas_features = betas[features_start : features_start + num_features]
        betas_interactions = betas[features_start + num_features :]
    else:
        betas_features = betas[features_start:]

    distance = agent_block_interaction(distance_interaction)(agent_coord, block_coords)

    # Build logit with controls (not interacted with distance)
    logit = beta_distance * distance

    # Add control terms (not interacted with distance)
    if num_controls > 0:
        logit = logit + (betas_controls * controls).sum(-1)

    # Add feature terms
    logit = logit + (betas_features * features).sum(-1)

    # Add interaction terms if enabled
    if interaction:
        logit = logit + jnp.einsum("f,c,cf->c", betas_interactions, distance, features)

    logsoftmax = jax.nn.log_softmax(logit)
    return -logsoftmax[chosen_block_id]


def generalized_chunked_sum(
    model_fn,
    in_axes,
    static_argnames=None,
):
    """
    Create a chunked sum version of any model function.

    Args:
        model_fn: The model function to chunk
        in_axes: Tuple/list specifying vmap in_axes for positional args
        static_argnames: List of kwarg names to treat as static in jit
        chunk_size: Size of each chunk

    Returns:
        A chunked version of the model that sums over samples
    """
    # Combine chunk_size with any existing static_argnames
    all_static_argnames = ["chunk_size"]
    if static_argnames:
        all_static_argnames.extend(static_argnames)

    @partial(jax.jit, static_argnames=all_static_argnames)
    def chunked_model(*args, chunk_size=1024, **kwargs):
        # Find which args need chunking (where in_axes[i] == 0)
        chunked_indices = [i for i, axis in enumerate(in_axes) if axis == 0]

        if not chunked_indices:
            # No chunking needed, just call the model
            return model_fn(*args, **kwargs)

        # Create a partially applied version of model_fn with the static kwargs
        # This avoids passing kwargs through vmap
        model_fn_partial = partial(model_fn, **kwargs)

        # Get the batch size from the first chunked argument
        first_chunked_idx = chunked_indices[0]
        num_samples = args[first_chunked_idx].shape[0]

        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        padded_size = num_chunks * chunk_size
        pad_size = padded_size - num_samples

        # Create mask for valid samples
        mask = jnp.arange(padded_size) < num_samples

        # Pad and chunk all necessary arguments
        args_list = list(args)
        chunked_args = []

        for idx in chunked_indices:
            arg = args_list[idx]
            # Determine padding shape based on argument dimensions
            if arg.ndim == 1:
                padded = jnp.pad(arg, (0, pad_size), mode="constant")
                chunked = padded.reshape(num_chunks, chunk_size)
            else:
                pad_width = [(0, pad_size)] + [(0, 0)] * (arg.ndim - 1)
                padded = jnp.pad(arg, pad_width, mode="constant")
                new_shape = (num_chunks, chunk_size) + arg.shape[1:]
                chunked = padded.reshape(new_shape)
            chunked_args.append(chunked)

        mask_chunked = mask.reshape(num_chunks, chunk_size)

        @jax.checkpoint
        def scan_fn(total_loss, chunk_data):
            # Unpack chunk data
            *chunk_args, chunk_mask = chunk_data

            # Reconstruct args with chunked values
            chunk_args_list = list(args)
            for i, idx in enumerate(chunked_indices):
                chunk_args_list[idx] = chunk_args[i]

            # Use the partially applied model function (no kwargs needed)
            chunk_losses = vmap(model_fn_partial, in_axes)(*chunk_args_list)

            # Apply mask and sum only real samples
            chunk_sum = jnp.sum(chunk_losses * chunk_mask)
            new_total = total_loss + chunk_sum

            return new_total, None

        # Run scan over chunks
        final_loss, _ = lax.scan(
            scan_fn,
            0.0,
            (*chunked_args, mask_chunked),
        )

        return final_loss

    return chunked_model


# Now you can create chunked versions of your models like this:

# For dcm_model - updated axes for new parameter order with single betas
dcm_model_chunked_sum = generalized_chunked_sum(
    dcm_model,
    in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None),
    static_argnames=["distance_interaction", "race_interaction", "income_interaction"],
)

# For dcm_model_interactions - updated axes for new parameter order with single betas and controls
dcm_model_interactions_chunked_sum = generalized_chunked_sum(
    dcm_model_interactions,
    in_axes=(None, 0, 0, None, None, None),  # Added one more None for controls
    static_argnames=[
        "distance_interaction",
        "distance_features_interaction",
        "interaction",
    ],
)


# Vectorized version that returns individual sample losses - updated axes
def create_dcm_model_samples():
    def dcm_model_samples(
        betas,
        agent_home_ids,
        agent_coords,
        agent_race_onehots,
        chosen_block_ids,
        chosen_block_coords,
        block_coords,
        block_race_dists,
        block_incomes,
        features,
        distance_interaction="l2_log",
        race_interaction="dissimilarity",
        income_interaction="abs_diff",
    ):
        # Create a partial function with the static kwargs
        model_fn = partial(
            dcm_model,
            distance_interaction=distance_interaction,
            race_interaction=race_interaction,
            income_interaction=income_interaction,
        )

        # Use vmap with the partial function
        return vmap(model_fn, (None, 0, 0, 0, 0, 0, None, None, None, None), 0)(
            betas,
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

    return jax.jit(
        dcm_model_samples,
        static_argnames=[
            "distance_interaction",
            "race_interaction",
            "income_interaction",
        ],
    )


dcm_model_samples = create_dcm_model_samples()


# Vectorized version for interactions model that returns individual sample losses
def create_dcm_model_interactions_samples():
    def dcm_model_interactions_samples(
        betas,
        agent_coords,
        chosen_block_ids,
        block_coords,
        controls,
        features,
        distance_interaction="l2_log",
        distance_features_interaction="l2_log_product",
        interaction=True,
    ):
        # Create a partial function with the static kwargs
        model_fn = partial(
            dcm_model_interactions,
            distance_interaction=distance_interaction,
            distance_features_interaction=distance_features_interaction,
            interaction=interaction,
        )

        # Use vmap with the partial function - updated axes to include controls
        return vmap(model_fn, (None, 0, 0, None, None, None), 0)(
            betas,
            agent_coords,
            chosen_block_ids,
            block_coords,
            controls,
            features,
        )

    return jax.jit(
        dcm_model_interactions_samples,
        static_argnames=[
            "distance_interaction",
            "distance_features_interaction",
            "interaction",
        ],
    )


dcm_model_interactions_samples = create_dcm_model_interactions_samples()


# ============================================================================
# NEW MODEL: Base model with network features and distance interactions
# ============================================================================

@partial(
    jax.jit,
    static_argnames=["distance_interaction", "race_interaction", "income_interaction", "network_interaction"],
)
def dcm_model_network_interactions(
    # All betas in one array
    betas,  # (3 + N_network + N_network_interaction + F_control)
    # Agent related
    agent_home_id,  # ()
    agent_coord,  # (2)
    agent_race_onehot,  # (R)
    chosen_block_id,  # ()
    chosen_block_coord,  # (2)
    # Block related
    block_coords,  # (C, 2)
    block_race_dists,  # (C, R)
    block_incomes,  # (C)
    # Network features (core research variables)
    network_features,  # (C, N_network)
    # Control features
    control_features,  # (C, F_control)
    # kwargs
    distance_interaction: str = "l2_log",
    race_interaction: str = "dissimilarity",
    income_interaction: str = "abs_diff",
    network_interaction: bool = True,  # Whether to include distance × network interactions
):
    """
    DCM model with network features as core variables.
    
    Model structure:
        logit = β_distance × distance
              + β_race × race_dissimilarity
              + β_income × income_difference
              + Σ β_network_k × network_feature_k        # Main effects
              + Σ β_interaction_k × (distance × network_feature_k)  # Interactions (if enabled)
              + Σ β_control_m × control_feature_m        # Controls
    
    Args:
        betas: Parameter vector with structure:
            [β_distance, β_race, β_income, 
             β_network_1, ..., β_network_N,
             β_interaction_1, ..., β_interaction_N (if network_interaction=True),
             β_control_1, ..., β_control_M]
        network_features: (C, N_network) - Core network features to be highlighted
        control_features: (C, F_control) - Control block features
        network_interaction: If True, include distance × network_feature interactions
    """
    # Unpack betas
    beta_distance = betas[0]
    beta_race = betas[1]
    beta_income = betas[2]
    
    num_network = network_features.shape[1]
    num_controls = control_features.shape[1]
    
    # Extract network feature betas (main effects)
    betas_network = betas[3:3+num_network]
    
    # Extract interaction betas if enabled
    if network_interaction:
        betas_interactions = betas[3+num_network:3+2*num_network]
        betas_controls = betas[3+2*num_network:]
    else:
        betas_controls = betas[3+num_network:]
    
    # Calculate distance (same logic as base model)
    distances = agent_block_interaction(distance_interaction)(
        agent_coord, block_coords
    )  # (C,)
    actual_distance = agent_agent_interaction(distance_interaction)(
        agent_coord, chosen_block_coord
    )  # ()
    condition = (agent_home_id == chosen_block_id) & (
        jnp.arange(block_coords.shape[0]) == agent_home_id
    )  # (C,)
    distance = jnp.where(condition, actual_distance, distances)  # (C,)
    
    # Calculate race and income dissimilarity
    if race_interaction == "threshold":
        race_diss = agent_block_interaction(race_interaction)(
            agent_race_onehot, block_race_dists
        )
    else:
        race_diss = block_block_interaction(race_interaction)(
            agent_home_id, block_race_dists
        )
    
    income_diss = block_block_interaction(income_interaction)(
        agent_home_id, block_incomes
    )
    
    # Build logit
    logit = (
        beta_distance * distance
        + beta_race * race_diss
        + beta_income * income_diss
        + (betas_network * network_features).sum(-1)  # Network main effects
    )
    
    # Add interaction terms if enabled
    if network_interaction:
        # distance × network_features interactions
        # Shape: (C,) = einsum over k: (k,) × (C,) × (C, k)
        logit = logit + jnp.einsum("k,c,ck->c", betas_interactions, distance, network_features)
    
    # Add control features
    if num_controls > 0:
        logit = logit + (betas_controls * control_features).sum(-1)
    
    logsoftmax = jax.nn.log_softmax(logit)
    return -logsoftmax[chosen_block_id]


# Chunked version for optimization
dcm_model_network_interactions_chunked_sum = generalized_chunked_sum(
    dcm_model_network_interactions,
    in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None, None),
    static_argnames=["distance_interaction", "race_interaction", "income_interaction", "network_interaction"],
)


# Vectorized version for computing individual sample losses
def create_dcm_model_network_interactions_samples():
    def dcm_model_network_interactions_samples(
        betas,
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
        distance_interaction="l2_log",
        race_interaction="dissimilarity",
        income_interaction="abs_diff",
        network_interaction=True,
    ):
        # Create a partial function with the static kwargs
        model_fn = partial(
            dcm_model_network_interactions,
            distance_interaction=distance_interaction,
            race_interaction=race_interaction,
            income_interaction=income_interaction,
            network_interaction=network_interaction,
        )
        
        # Use vmap
        return vmap(model_fn, (None, 0, 0, 0, 0, 0, None, None, None, None, None), 0)(
            betas,
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
    
    return jax.jit(
        dcm_model_network_interactions_samples,
        static_argnames=[
            "distance_interaction",
            "race_interaction",
            "income_interaction",
            "network_interaction",
        ],
    )


dcm_model_network_interactions_samples = create_dcm_model_network_interactions_samples()
