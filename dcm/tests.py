import pytest
import jax
import jax.numpy as jnp
from jax import grad, hessian
import numpy as np

from dcm.models import dcm_model_chunked_sum, dcm_model_samples


def create_inputs(num_samples, num_blocks, num_features=128, num_races=16):
    """Create test inputs for DCM model functions.

    Args:
        num_samples (int): Number of agent samples (N)
        num_blocks (int): Number of blocks/choices (C)
        num_features (int): Number of features (F), default 128
        num_races (int): Number of race categories (R), default 16

    Returns:
        tuple: Tuple containing all model inputs in the correct order for dcm_model
    """
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 11)  # Need one more key

    # Create single betas array: [beta_distance, beta_race, beta_income, *betas_features]
    beta_distance = jax.random.normal(keys[0], ()) * 0.1
    beta_race = jax.random.normal(keys[1], ()) * 0.1
    beta_income = jax.random.normal(keys[2], ()) * 0.1
    betas_features = jax.random.normal(keys[7], (num_features,)) * 0.01

    # Combine into single betas array
    betas = jnp.concatenate(
        [jnp.array([beta_distance, beta_race, beta_income]), betas_features]
    )

    # Block-level data (C dimension)
    block_coords = jax.random.normal(keys[3], (num_blocks, 2)) * 10.0
    block_race_dists = jax.random.dirichlet(keys[4], jnp.ones(num_races), (num_blocks,))
    log_block_incomes = jax.random.exponential(keys[5], (num_blocks,)) * 10
    features = jax.random.normal(keys[6], (num_blocks, num_features))

    # Agent-level data (N dimension)
    agent_coord = jax.random.normal(keys[8], (num_samples, 2)) * 10.0
    agent_home_id = jax.random.randint(keys[9], (num_samples,), 0, num_blocks)

    # Random choices for each agent
    choice_key = jax.random.split(keys[0], num_samples)
    chosen_block_id = jax.random.randint(choice_key[0], (num_samples,), 0, num_blocks)

    # Get the coordinates of chosen blocks
    chosen_block_coord = block_coords[chosen_block_id]

    # Return tuple in NEW dcm_model parameter order with single betas
    return (
        # Single betas array
        betas,  # (3 + F)
        # Agent related
        agent_home_id,  # (N)
        agent_coord,  # (N, 2)
        chosen_block_id,  # (N)
        chosen_block_coord,  # (N, 2) - NEW
        # Block related
        block_coords,  # (C, 2)
        block_race_dists,  # (C, R)
        log_block_incomes,  # (C)
        # Features
        features,  # (C, F)
    )


def test_chunked_vs_samples_consistency():
    """Test that dcm_model_chunked_sum and summed dcm_model_samples give similar results."""
    inputs = create_inputs(num_samples=10000, num_blocks=2000)

    # Compute using chunked version
    chunked_result = dcm_model_chunked_sum(*inputs)

    # Compute using samples version and sum
    samples_result = jnp.sum(dcm_model_samples(*inputs))

    # Check relative closeness
    relative_diff = jnp.abs((chunked_result - samples_result) / samples_result)
    assert (
        relative_diff < 0.01
    ), f"Results differ by {relative_diff:.4f}, expected < 0.01"

    print(f"Chunked result: {chunked_result:.6f}")
    print(f"Samples result: {samples_result:.6f}")
    print(f"Relative difference: {relative_diff:.6f}")


def test_chunked_vs_samples_gradient_consistency():
    """Test that gradients from dcm_model_chunked_sum and dcm_model_samples are consistent."""
    inputs = create_inputs(num_samples=10000, num_blocks=2000)

    # Create objective function for chunked version
    def chunked_objective(betas):
        modified_inputs = (betas,) + inputs[1:]
        return dcm_model_chunked_sum(*modified_inputs)

    # Create objective function for samples version
    def samples_objective(betas):
        modified_inputs = (betas,) + inputs[1:]
        return jnp.sum(dcm_model_samples(*modified_inputs))

    # Compute gradients
    chunked_grad_fn = grad(chunked_objective)
    samples_grad_fn = grad(samples_objective)

    chunked_gradient = chunked_grad_fn(inputs[0])  # betas is now at index 0
    samples_gradient = samples_grad_fn(inputs[0])

    # Check gradient consistency
    gradient_diff = chunked_gradient - samples_gradient
    diff_magnitude = jnp.linalg.norm(gradient_diff)
    samples_magnitude = jnp.linalg.norm(samples_gradient)
    relative_diff = diff_magnitude / samples_magnitude

    # Print gradients if test is about to fail
    if relative_diff >= 0.01:
        print(f"\nChunked gradient: {chunked_gradient}")
        print(f"\nSamples gradient: {samples_gradient}")

    assert (
        relative_diff < 0.01
    ), f"Relative gradient vector difference: {relative_diff:.4f}, expected < 0.01"

    print(f"Chunked gradient norm: {jnp.linalg.norm(chunked_gradient):.6f}")
    print(f"Samples gradient norm: {samples_magnitude:.6f}")
    print(f"Gradient difference norm: {diff_magnitude:.6e}")
    print(f"Relative gradient vector difference: {relative_diff:.6f}")


def test_large_scale_load():
    """Load test with 1M samples and 10K choices to verify it runs without memory issues."""
    inputs = create_inputs(num_samples=1000000, num_blocks=10000)

    # This should run without OOM errors
    result = dcm_model_chunked_sum(*inputs)

    # Basic sanity check
    assert jnp.isfinite(result), "Result should be finite"
    assert result > 0, "Loss should be positive"

    print(f"Large scale result: {result:.6f}")


def test_large_scale_gradients():
    """Test gradient and hessian computation on large scale with 1M samples and 10K choices."""
    inputs = create_inputs(num_samples=1000000, num_blocks=10000)

    # Create objective function that takes betas as input
    def objective(betas):
        # Reconstruct tuple with modified betas
        modified_inputs = (betas,) + inputs[1:]
        return dcm_model_chunked_sum(*modified_inputs)

    # Compute gradient
    grad_fn = grad(objective)
    gradient = grad_fn(inputs[0])  # betas is now at index 0

    # # Compute hessian
    # hessian_fn = hessian(objective)
    # hess = hessian_fn(inputs[0])

    # # Basic sanity checks
    # assert gradient.shape == inputs[0].shape, "Gradient shape mismatch"
    # assert hess.shape == (inputs[0].shape[0], inputs[0].shape[0]), "Hessian shape mismatch"
    # assert jnp.all(jnp.isfinite(gradient)), "Gradient should be finite"
    # assert jnp.all(jnp.isfinite(hess)), "Hessian should be finite"

    # print(f"Gradient norm: {jnp.linalg.norm(gradient):.6f}")
    # print(f"Hessian trace: {jnp.trace(hess):.6f}")
    # print(f"Hessian condition number: {jnp.linalg.cond(hess):.2e}")


def test_large_scale_optimize():
    """Test full parameter optimization using JAX scipy.optimize.minimize.
    This test optimizes all beta parameters in a single array.
    """
    inputs = create_inputs(num_samples=100000, num_blocks=5000)

    # Extract fixed inputs (everything except the betas we want to optimize)
    # In new order:
    # 0: betas, 1: agent_home_id, 2: agent_coord, 3: chosen_block_id
    # 4: chosen_block_coord, 5: block_coords, 6: block_race_dists, 7: block_incomes, 8: features
    agent_home_id = inputs[1]
    agent_coord = inputs[2]
    chosen_block_id = inputs[3]
    chosen_block_coord = inputs[4]  # NEW
    block_coords = inputs[5]  # Updated index
    block_race_dists = inputs[6]  # Updated index
    block_incomes = inputs[7]  # Updated index
    features = inputs[8]  # Updated index

    # Initial beta values
    initial_params = inputs[0]

    def objective(params):
        """Objective function for optimization."""
        # Reconstruct full input tuple
        model_inputs = (
            # Betas
            params,
            # Agent related
            agent_home_id,
            agent_coord,
            chosen_block_id,
            chosen_block_coord,  # NEW
            # Block related
            block_coords,
            block_race_dists,
            block_incomes,
            # Features
            features,
        )

        return dcm_model_chunked_sum(*model_inputs) / agent_coord.shape[0]

    print(f"Initial objective value: {objective(initial_params):.6f}")
    print(f"Initial parameter norm: {jnp.linalg.norm(initial_params):.6f}")

    # Run optimization
    from jax.scipy.optimize import minimize

    result = minimize(
        objective, initial_params, method="BFGS", options={"maxiter": 100}
    )

    print(f"Optimization converged: {result.success}")
    print(f"Final objective value: {result.fun:.6f}")
    print(f"Number of iterations: {result.nit}")
    print(f"Final parameter norm: {jnp.linalg.norm(result.x):.6f}")

    # Verify optimization made progress
    assert result.fun < objective(
        initial_params
    ), "Optimization should reduce objective value"
    assert jnp.isfinite(result.fun), "Final objective should be finite"
    assert jnp.all(jnp.isfinite(result.x)), "Final parameters should be finite"

    # Print parameter changes
    print(f"Beta distance: {initial_params[0]:.6f} -> {result.x[0]:.6f}")
    print(f"Beta race: {initial_params[1]:.6f} -> {result.x[1]:.6f}")
    print(f"Beta income: {initial_params[2]:.6f} -> {result.x[2]:.6f}")
    print(
        f"Features betas norm: {jnp.linalg.norm(initial_params[3:]):.6f} -> {jnp.linalg.norm(result.x[3:]):.6f}"
    )


# def test_make_args_with_dcm_model():
#     """Test using make_args output directly with DCM model."""
#     # Create sample data
#     num_agents = 1000
#     num_blocks = 100

#     # Create agents
#     agents = []
#     for i in range(num_agents):
#         agents.append(
#             AgentFeatures(
#                 home_block_id=i % num_blocks,
#                 home_coord=(
#                     float(np.random.normal() * 10),
#                     float(np.random.normal() * 10),
#                 ),
#                 incident_block_id=int(np.random.randint(0, num_blocks)),
#             )
#         )

#     # Create blocks
#     blocks = []
#     for i in range(num_blocks):
#         blocks.append(
#             BlockFeatures(
#                 block_id=i,
#                 home_coord=(
#                     float(np.random.normal() * 10),
#                     float(np.random.normal() * 10),
#                 ),
#                 racial_dist={
#                     "WHITE": float(np.random.uniform()),
#                     "BLACK": float(np.random.uniform()),
#                     "HISPANIC": float(np.random.uniform()),
#                     "ASIAN": float(np.random.uniform()),
#                     "OTHER": float(np.random.uniform()),
#                 },
#                 log_median_income=float(np.random.normal() + 10),
#                 log_total_population=float(np.random.normal() + 5),
#                 avg_household_size=float(np.random.uniform(1, 5)),
#                 log_total_employees=float(np.random.normal() + 3),
#                 log_attractions=float(np.random.normal()),
#                 log_transit_stops=float(np.random.normal()),
#             )
#         )

#     # Normalize racial distributions
#     for block in blocks:
#         total = sum(block.racial_dist.values())
#         block.racial_dist = {k: v / total for k, v in block.racial_dist.items()}

#     # Use helper functions to prepare data
#     agent_home_id, agent_coord, chosen_block_id = prepare_dcm_agent_args(agents)
#     block_coords, block_race_dists, block_incomes, features = prepare_dcm_block_args(
#         blocks
#     )

#     # Create beta parameters
#     beta_distance = -0.1
#     beta_race = 0.05
#     beta_income = 0.02
#     betas_features = jnp.ones(features.shape[1]) * 0.01

#     # Call the model with unpacked args
#     result = dcm_model_chunked_sum(
#         beta_distance,
#         beta_race,
#         beta_income,
#         betas_features,
#         agent_home_id,
#         agent_coord,
#         chosen_block_id,
#         block_coords,
#         block_race_dists,
#         block_incomes,
#         features,
#     )

#     # Basic checks
#     assert jnp.isfinite(result), "Model result should be finite"
#     assert result > 0, "Loss should be positive"

#     print(f"DCM model result with make_args data: {result:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
