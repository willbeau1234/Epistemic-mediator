"""
TASK 2: Heuristic Agent Experiment - Break Simulation Circularity

Implements two heuristic agents that do NOT use Equations 30-31:
- Agent A (Stubborn): after friction, moves 0.1 further toward current hypothesis
- Agent B (Flexible): after friction, accepts friction-corrected prior with 80% probability

Tests whether Belief Versioning can correctly identify behavioral types
without encoding the theoretical model.
"""

import sys
import io
import json
import numpy as np
from datetime import datetime

print("="*80)
print("TASK 2: HEURISTIC AGENT EXPERIMENT")
print("="*80)
print()

# Suppress loading prints
print("Loading delusion2.py...")
old_stdout = sys.stdout
sys.stdout = io.StringIO()

from delusion2 import (
    H, ur_prior, human, bot, do_sample_from_prior, world_model,
    obs_val_space, compute_kl_divergence_jit, extract_final_beliefs_batch
)
import jax
import jax.numpy as jnp

sys.stdout = old_stdout
print("✓ Loaded\n")

# Configuration matching main experiments
NUM_SIMS = 1000
TIME_HORIZON = 50
P_CHI = 90  # pχ = 0.9
SEED = 42
P_TRUE = 0.5
HUMAN_LEVEL = 0
BOT_LEVEL = 0
HONEST = False
UNIFORM = False

# Belief versioning parameters (matching main experiments)
TAU_V = 0.02
TAU_H = -0.05
ENTROPY_MIN = 1.0
VELOCITY_EPSILON = 0.02
BELIEF_DELTA = 0.3
TYPE_CONFIDENCE_THRESHOLD = 0.7
FRICTION_BETA = 1.5

print("Configuration:")
print(f"  N={NUM_SIMS}, T={TIME_HORIZON}, pχ={P_CHI}, seed={SEED}")
print(f"  Matching main experiment parameters")
print()

# =============================================================================
# HEURISTIC AGENT IMPLEMENTATIONS
# =============================================================================

@jax.jit
def agent_a_stubborn_response(
    prior: jnp.ndarray,
    bayes_prior: jnp.ndarray,
    friction_applied_prior: jnp.ndarray,
    current_belief: float,
    friction: float
) -> jnp.ndarray:
    """
    Agent A (Stubborn): After any friction event, moves belief 0.1 FURTHER
    toward current hypothesis regardless of evidence.

    This is DIFFERENT from Equation 31 (θ_V resistance formula).
    """
    # If no friction, accept Bayesian update
    no_friction_case = bayes_prior

    # If friction > 0, RESIST by moving 0.1 further toward current hypothesis
    # Current hypothesis is whichever direction belief leans (>0.5 or <0.5)
    stubborn_direction = jnp.where(current_belief > 0.5, 0.1, -0.1)

    # Compute target belief after stubbornness
    target_belief = jnp.clip(current_belief + stubborn_direction, 0.01, 0.99)

    # Blend prior to achieve target belief
    # This is a heuristic that moves the entire prior distribution
    # to approximately match the target marginal belief
    p_h1_friction = friction_applied_prior[H.H1].sum() / (friction_applied_prior.sum() + 1e-10)

    # If target is higher than friction-applied belief, boost H1
    # If lower, boost H0
    adjustment = target_belief - p_h1_friction

    # Simple adjustment: scale H1 vs H0 probabilities
    adjusted_prior = jnp.where(
        friction > 0,
        friction_applied_prior.at[H.H1].multiply(1.0 + adjustment).at[H.H0].multiply(1.0 - adjustment),
        friction_applied_prior
    )

    # Normalize
    adjusted_prior = adjusted_prior / (adjusted_prior.sum() + 1e-10)

    return jnp.where(friction > 0, adjusted_prior, no_friction_case)


@jax.jit
def agent_b_flexible_response(
    prior: jnp.ndarray,
    bayes_prior: jnp.ndarray,
    friction_applied_prior: jnp.ndarray,
    friction: float,
    random_key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Agent B (Flexible): After any friction event, accepts friction-corrected
    prior with 80% probability, otherwise sticks with Bayesian update.

    This is DIFFERENT from Equation 30 (θ_G acceptance formula).
    """
    # If no friction, accept Bayesian update
    no_friction_case = bayes_prior

    # If friction > 0, accept friction-applied prior with 80% probability
    accept_friction = jax.random.bernoulli(random_key, p=0.8)

    flexible_prior = jnp.where(
        accept_friction,
        friction_applied_prior,  # Accept the friction (80%)
        bayes_prior              # Reject and keep Bayesian (20%)
    )

    return jnp.where(friction > 0, flexible_prior, no_friction_case)


# =============================================================================
# SIMULATION WITH HEURISTIC AGENTS
# =============================================================================

def run_heuristic_agent_simulation(
    agent_type: str,  # 'stubborn' or 'flexible'
    num_sims: int = NUM_SIMS,
    time_horizon: int = TIME_HORIZON,
    seed: int = SEED
):
    """
    Run Belief Versioning simulation with heuristic agent.

    Returns: (priors, frictions, type_confidences, checkouts, works, agent_labels)
    """
    h_world = H.H1
    p_chi = P_CHI

    # Pre-compute priors
    prior = ur_prior(P_TRUE, 1, 1)
    uncertainty_prior = ur_prior(0.5, 1, 1)

    init_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)
    init_entropy = -jnp.sum(
        jnp.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0) *
        jnp.log(jnp.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0))
    )
    belief_hist = jnp.ones(10) * init_belief
    entropy_hist = jnp.ones(10) * init_entropy

    commit_beliefs = jnp.zeros((10,) + prior.shape)
    commit_beliefs = commit_beliefs.at[0].set(prior)

    def step(carry, _t):
        (prior_state, key, belief_hist, entropy_hist,
         commit_beliefs, commit_count, type_confidence,
         friction_response_sum, friction_response_count,
         last_friction, last_belief, cumulative_work) = carry

        key, key_user, key_world, key_bot, key_agent = jax.random.split(key, num=5)

        # Human ventures opinion
        h_human = jax.random.choice(key_user, jnp.array([H.H0, H.H1]),
                                    p=do_sample_from_prior(prior_state, HONEST))

        # World generates data
        from delusion2 import Data
        d = jax.random.choice(key_world, jnp.array(list(Data)),
                             p=world_model()[h_world])

        # Current belief
        current_belief = prior_state[H.H1].sum() / (prior_state.sum() + 1e-10)

        # Entropy
        p_norm = prior_state / (prior_state.sum() + 1e-10)
        p_norm = jnp.clip(p_norm, 1e-10, 1.0)
        belief_entropy = -jnp.sum(p_norm * jnp.log(p_norm))

        # Update histories
        belief_hist = jnp.roll(belief_hist, -1).at[-1].set(current_belief)
        entropy_hist = jnp.roll(entropy_hist, -1).at[-1].set(belief_entropy)

        # Compute velocities
        v_e = jnp.mean(jnp.diff(belief_hist[-4:]))
        delta_h = jnp.mean(jnp.diff(entropy_hist[-4:]))

        # Commit if healthy
        commit_condition = (
            (belief_entropy > ENTROPY_MIN) &
            (jnp.abs(v_e) < VELOCITY_EPSILON) &
            (current_belief > BELIEF_DELTA) &
            (current_belief < (1 - BELIEF_DELTA))
        )
        commit_idx = commit_count % 10
        commit_beliefs = jnp.where(
            commit_condition,
            commit_beliefs.at[commit_idx].set(prior_state),
            commit_beliefs
        )
        commit_count = jnp.where(commit_condition, commit_count + 1, commit_count)

        # Type classification (same as original versioning)
        expected_after_friction = (1 - last_friction) * last_belief + last_friction * 0.5
        expected_distance = jnp.abs(expected_after_friction - 0.5)
        actual_distance = jnp.abs(current_belief - 0.5)

        is_validation_response = (last_friction > 0) & (actual_distance > expected_distance + 0.01)

        friction_response_sum = jnp.where(
            last_friction > 0,
            friction_response_sum + is_validation_response,
            friction_response_sum
        )
        friction_response_count = jnp.where(
            last_friction > 0,
            friction_response_count + 1,
            friction_response_count
        )
        type_confidence = (friction_response_sum + 1) / (friction_response_count + 2)

        # Trigger and friction
        triggered = (v_e > TAU_V) & (delta_h < TAU_H)
        base_friction = jnp.where(triggered, 0.3, 0.0)

        is_validation_type = type_confidence > TYPE_CONFIDENCE_THRESHOLD
        friction = jnp.where(
            is_validation_type,
            FRICTION_BETA * base_friction,
            base_friction
        )

        # Checkout
        checkout_triggered = is_validation_type & triggered & (commit_count > 0)

        # Bot responds
        obs, val = jax.random.choice(
            key_bot, obs_val_space,
            p=bot(prior=prior_state, level=BOT_LEVEL, honest=HONEST, uniform=UNIFORM)[
                p_chi, h_human, h_world, d
            ].reshape(-1)
        )

        # Bayesian update
        bayes_prior = human(prior=prior_state, level=HUMAN_LEVEL, honest=HONEST, uniform=UNIFORM)[
            h_human, obs, val
        ]

        # Friction-applied prior
        friction_applied_prior = (1 - friction) * bayes_prior + friction * uncertainty_prior

        # === HEURISTIC AGENT BEHAVIOR (REPLACES EQUATIONS 30-31) ===
        if agent_type == 'stubborn':
            new_prior = agent_a_stubborn_response(
                prior_state, bayes_prior, friction_applied_prior,
                current_belief, friction
            )
        else:  # flexible
            new_prior = agent_b_flexible_response(
                prior_state, bayes_prior, friction_applied_prior,
                friction, key_agent
            )

        # Checkout
        last_commit_idx = (commit_count - 1) % 10
        checkout_prior = commit_beliefs[last_commit_idx]
        new_prior = jnp.where(
            checkout_triggered,
            checkout_prior,
            new_prior
        )
        type_confidence = jnp.where(checkout_triggered, 0.5, type_confidence)
        friction_response_sum = jnp.where(checkout_triggered, 0.0, friction_response_sum)
        friction_response_count = jnp.where(checkout_triggered, 0.0, friction_response_count)

        # Epistemic work
        work = compute_kl_divergence_jit(new_prior.flatten(), prior_state.flatten())
        cumulative_work = cumulative_work + work

        new_carry = (
            new_prior, key, belief_hist, entropy_hist,
            commit_beliefs, commit_count, type_confidence,
            friction_response_sum, friction_response_count,
            friction, current_belief, cumulative_work
        )

        outputs = (new_prior, friction, type_confidence, checkout_triggered, work)
        return new_carry, outputs

    def run_one_sim(seed_val):
        key = jax.random.key(seed_val)

        init_carry = (
            prior,
            key,
            belief_hist,
            entropy_hist,
            commit_beliefs,
            1,      # commit_count
            0.5,    # type_confidence
            0.0,    # friction_response_sum
            0.0,    # friction_response_count
            0.0,    # last_friction
            init_belief,  # last_belief
            0.0     # cumulative_work
        )

        outputs = jax.lax.scan(
            f=step,
            init=init_carry,
            length=time_horizon
        )[1]

        return outputs

    print(f"  Compiling and running {agent_type} agent simulation...")
    results = jax.lax.map(run_one_sim, jnp.arange(num_sims), batch_size=1000)

    priors, frictions, type_confs, checkouts, works = results

    # Agent label (1 for stubborn/validation-like, 0 for flexible/growth-like)
    agent_label = 1 if agent_type == 'stubborn' else 0

    return priors, frictions, type_confs, checkouts, works, agent_label


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

print("="*80)
print("RUNNING HEURISTIC AGENT EXPERIMENTS")
print("="*80)
print()

# Agent A: Stubborn
print("[1/2] Agent A (Stubborn)")
print("-" * 40)
results_stubborn = run_heuristic_agent_simulation('stubborn')
priors_a, frictions_a, type_conf_a, checkouts_a, works_a, label_a = results_stubborn

final_beliefs_a = extract_final_beliefs_batch(priors_a)
spiral_rate_a = float((final_beliefs_a > 0.9).mean())
final_type_conf_a = type_conf_a[:, -1]
total_work_a = works_a.sum(axis=1)

# Classification: system thinks stubborn is validation-seeking?
detected_validation_a = (final_type_conf_a > TYPE_CONFIDENCE_THRESHOLD).sum()
detection_rate_a = float(detected_validation_a / NUM_SIMS)

print(f"  Spiral rate: {spiral_rate_a:.1%}")
print(f"  Mean final type confidence: {final_type_conf_a.mean():.3f}")
print(f"  Detected as validation-seeking: {detection_rate_a:.1%}")
print(f"  Mean epistemic work: {total_work_a.mean():.3f}")
print()

# Agent B: Flexible
print("[2/2] Agent B (Flexible)")
print("-" * 40)
results_flexible = run_heuristic_agent_simulation('flexible')
priors_b, frictions_b, type_conf_b, checkouts_b, works_b, label_b = results_flexible

final_beliefs_b = extract_final_beliefs_batch(priors_b)
spiral_rate_b = float((final_beliefs_b > 0.9).mean())
final_type_conf_b = type_conf_b[:, -1]
total_work_b = works_b.sum(axis=1)

# Classification: system thinks flexible is growth-seeking?
detected_growth_b = (final_type_conf_b < (1 - TYPE_CONFIDENCE_THRESHOLD)).sum()
detection_rate_b = float(detected_growth_b / NUM_SIMS)

print(f"  Spiral rate: {spiral_rate_b:.1%}")
print(f"  Mean final type confidence: {final_type_conf_b.mean():.3f}")
print(f"  Detected as growth-seeking: {detection_rate_b:.1%}")
print(f"  Mean epistemic work: {total_work_b.mean():.3f}")
print()

# =============================================================================
# ANALYSIS: DOES VERSIONING CORRECTLY IDENTIFY TYPES?
# =============================================================================

print("="*80)
print("ANALYSIS: TYPE DETECTION ACCURACY")
print("="*80)
print()

# Ground truth:
# - Agent A (Stubborn) should be detected as validation-seeking (type_conf > 0.7)
# - Agent B (Flexible) should be detected as growth-seeking (type_conf < 0.3)

true_positive_a = float((final_type_conf_a > TYPE_CONFIDENCE_THRESHOLD).sum())
true_positive_b = float((final_type_conf_b < (1 - TYPE_CONFIDENCE_THRESHOLD)).sum())

total = NUM_SIMS * 2
correct = true_positive_a + true_positive_b
accuracy = correct / total

print(f"Agent A (Stubborn - should be validation-seeking):")
print(f"  Correctly identified: {true_positive_a}/{NUM_SIMS} ({true_positive_a/NUM_SIMS:.1%})")
print()

print(f"Agent B (Flexible - should be growth-seeking):")
print(f"  Correctly identified: {true_positive_b}/{NUM_SIMS} ({true_positive_b/NUM_SIMS:.1%})")
print()

print(f"Overall Accuracy: {accuracy:.1%}")
print(f"Chance Level: 50.0%")
print()

# Statistical test: binomial test
from scipy.stats import binomtest
p_value = binomtest(int(correct), total, p=0.5, alternative='greater').pvalue

print(f"Binomial test (H0: accuracy = chance):")
print(f"  p-value: {p_value:.2e}")
if p_value < 0.05:
    print(f"  ✓ Significantly above chance (p < 0.05)")
else:
    print(f"  ✗ Not significantly above chance")
print()

# =============================================================================
# VERDICT
# =============================================================================

print("="*80)
print("VERDICT: CIRCULARITY BROKEN?")
print("="*80)
print()

if accuracy > 0.55 and p_value < 0.05:
    print("✓ YES - Belief Versioning correctly identifies behavioral types")
    print("  even when agents do NOT use Equations 30-31")
    print()
    print("  This demonstrates the mechanism detects BEHAVIORAL patterns")
    print("  (resistance vs acceptance of friction) rather than just")
    print("  matching the programmed theoretical model.")
    print()
    print("  The simulation is NOT circular - it shows the mechanism can")
    print("  infer hidden types from observable behavior.")
else:
    print("✗ NO - Detection is not significantly above chance")
    print()
    print("  This suggests the mechanism may only work when agents")
    print("  are explicitly programmed according to Equations 30-31.")
    print()
    print("  The circularity concern is validated.")

print()

# =============================================================================
# SAVE RESULTS
# =============================================================================

output = {
    'timestamp': datetime.now().isoformat(),
    'task': 'heuristic_agent_experiment',
    'configuration': {
        'num_sims': NUM_SIMS,
        'time_horizon': TIME_HORIZON,
        'p_chi': P_CHI,
        'seed': SEED,
    },
    'agent_a_stubborn': {
        'spiral_rate': spiral_rate_a,
        'mean_type_confidence': float(final_type_conf_a.mean()),
        'detected_as_validation': detection_rate_a,
        'mean_epistemic_work': float(total_work_a.mean()),
        'correctly_identified': float(true_positive_a / NUM_SIMS)
    },
    'agent_b_flexible': {
        'spiral_rate': spiral_rate_b,
        'mean_type_confidence': float(final_type_conf_b.mean()),
        'detected_as_growth': detection_rate_b,
        'mean_epistemic_work': float(total_work_b.mean()),
        'correctly_identified': float(true_positive_b / NUM_SIMS)
    },
    'overall': {
        'accuracy': accuracy,
        'chance_level': 0.5,
        'p_value': float(p_value),
        'significantly_above_chance': bool(p_value < 0.05),
        'circularity_broken': bool(accuracy > 0.55 and p_value < 0.05)
    }
}

with open('heuristic_agent_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("✓ Saved to heuristic_agent_results.json")
print()

print("="*80)
print("TASK 2 COMPLETE")
print("="*80)
