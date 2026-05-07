"""
Task 2B: Test model-agnostic detector on agents with CLEAN directional signatures

Agent C (Doubler): Always moves belief AWAY from 0.5 after friction
Agent D (Moderator): Always moves belief TOWARD 0.5 after friction

These agents should be easily detectable (~90%+ accuracy) because they produce
the exact signatures the model-agnostic detector looks for.

This creates a positive control: detector works on directional agents,
fails on adversarial/noisy agents (Stubborn/Flexible from task2).
"""

import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
import json
from datetime import datetime

# Import from delusion2
from delusion2 import H

print("="*80)
print("TASK 2B: DIRECTIONAL AGENT EXPERIMENT")
print("="*80)
print()

# Configuration matching main experiment
NUM_SIMS = 1000
TIME_HORIZON = 50
P_CHI = 90
SEED = 42
TYPE_CONFIDENCE_THRESHOLD = 0.7

# Agent parameters
FRICTION_BETA = 0.8
TAU_V = 0.01
TAU_H = -0.01
ENTROPY_MIN = 0.5
VELOCITY_EPSILON = 0.05
BELIEF_DELTA = 0.1
HONEST = 0.9

@jax.jit
def agent_c_doubler_response(
    prior: jnp.ndarray,
    bayes_prior: jnp.ndarray,
    friction_applied_prior: jnp.ndarray,
    friction: float,
    random_key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Agent C: DOUBLER

    Always moves belief AWAY from 0.5 after friction.
    This should trigger θ_V (validation-seeking) classification.

    Strategy:
    - Compute current belief
    - Move it 0.15 further from 0.5
    - Adjust prior to match target belief
    """
    # If no friction, just return Bayes update
    no_friction_case = bayes_prior

    # Current belief from friction-applied prior
    current_belief = friction_applied_prior[H.H1].sum() / (friction_applied_prior.sum() + 1e-10)

    # Move AWAY from 0.5
    if current_belief > 0.5:
        target_belief = jnp.clip(current_belief + 0.15, 0.5, 0.99)
    else:
        target_belief = jnp.clip(current_belief - 0.15, 0.01, 0.5)

    # Adjust prior to match target
    p_h1_current = friction_applied_prior[H.H1].sum() / (friction_applied_prior.sum() + 1e-10)
    adjustment = target_belief - p_h1_current

    adjusted_prior = jnp.where(
        friction > 0,
        friction_applied_prior.at[H.H1].multiply(1.0 + adjustment).at[H.H0].multiply(1.0 - adjustment),
        friction_applied_prior
    )

    # Normalize
    adjusted_prior = adjusted_prior / (adjusted_prior.sum() + 1e-10)

    return jnp.where(friction > 0, adjusted_prior, no_friction_case)


@jax.jit
def agent_d_moderator_response(
    prior: jnp.ndarray,
    bayes_prior: jnp.ndarray,
    friction_applied_prior: jnp.ndarray,
    friction: float,
    random_key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Agent D: MODERATOR

    Always moves belief TOWARD 0.5 after friction.
    This should trigger θ_G (growth-seeking) classification.

    Strategy:
    - Compute current belief
    - Move it 0.15 closer to 0.5
    - Adjust prior to match target belief
    """
    # If no friction, just return Bayes update
    no_friction_case = bayes_prior

    # Current belief from friction-applied prior
    current_belief = friction_applied_prior[H.H1].sum() / (friction_applied_prior.sum() + 1e-10)

    # Move TOWARD 0.5
    if current_belief > 0.5:
        target_belief = jnp.clip(current_belief - 0.15, 0.5, 0.99)
    else:
        target_belief = jnp.clip(current_belief + 0.15, 0.01, 0.5)

    # Adjust prior to match target
    p_h1_current = friction_applied_prior[H.H1].sum() / (friction_applied_prior.sum() + 1e-10)
    adjustment = target_belief - p_h1_current

    adjusted_prior = jnp.where(
        friction > 0,
        friction_applied_prior.at[H.H1].multiply(1.0 + adjustment).at[H.H0].multiply(1.0 - adjustment),
        friction_applied_prior
    )

    # Normalize
    adjusted_prior = adjusted_prior / (adjusted_prior.sum() + 1e-10)

    return jnp.where(friction > 0, adjusted_prior, no_friction_case)


# Import the simulation function from task2
from task2_heuristic_agents import run_heuristic_agent_simulation, extract_final_beliefs_batch

print("Configuration:")
print(f"  N={NUM_SIMS}, T={TIME_HORIZON}, pχ={P_CHI}, seed={SEED}")
print()

print("="*80)
print("RUNNING DIRECTIONAL AGENT EXPERIMENTS")
print("="*80)
print()

# Agent C: Doubler
print("[1/2] Agent C (Doubler - moves away from 0.5)")
print("-" * 40)
results_doubler = run_heuristic_agent_simulation('doubler', agent_c_doubler_response)
priors_c, frictions_c, type_conf_c, checkouts_c, works_c, label_c = results_doubler

final_beliefs_c = extract_final_beliefs_batch(priors_c)
spiral_rate_c = float((final_beliefs_c > 0.9).mean())
final_type_conf_c = type_conf_c[:, -1]
total_work_c = works_c.sum(axis=1)

# Detection: Doubler should be detected as validation-seeking (type_conf > 0.7)
detected_validation_c = (final_type_conf_c > TYPE_CONFIDENCE_THRESHOLD).sum()
detection_rate_c = float(detected_validation_c / NUM_SIMS)

print(f"  Spiral rate: {spiral_rate_c:.1%}")
print(f"  Mean final type confidence: {final_type_conf_c.mean():.3f}")
print(f"  Detected as validation-seeking: {detection_rate_c:.1%}")
print(f"  Mean epistemic work: {total_work_c.mean():.3f}")
print()

# Agent D: Moderator
print("[2/2] Agent D (Moderator - moves toward 0.5)")
print("-" * 40)
results_moderator = run_heuristic_agent_simulation('moderator', agent_d_moderator_response)
priors_d, frictions_d, type_conf_d, checkouts_d, works_d, label_d = results_moderator

final_beliefs_d = extract_final_beliefs_batch(priors_d)
spiral_rate_d = float((final_beliefs_d > 0.9).mean())
final_type_conf_d = type_conf_d[:, -1]
total_work_d = works_d.sum(axis=1)

# Detection: Moderator should be detected as growth-seeking (type_conf < 0.3)
detected_growth_d = (final_type_conf_d < (1 - TYPE_CONFIDENCE_THRESHOLD)).sum()
detection_rate_d = float(detected_growth_d / NUM_SIMS)

print(f"  Spiral rate: {spiral_rate_d:.1%}")
print(f"  Mean final type confidence: {final_type_conf_d.mean():.3f}")
print(f"  Detected as growth-seeking: {detection_rate_d:.1%}")
print(f"  Mean epistemic work: {total_work_d.mean():.3f}")
print()

# Analysis
print("="*80)
print("ANALYSIS: TYPE DETECTION ACCURACY")
print("="*80)
print()

# Ground truth:
# - Agent C (Doubler) should be detected as validation-seeking (type_conf > 0.7)
# - Agent D (Moderator) should be detected as growth-seeking (type_conf < 0.3)

true_positive_c = float((final_type_conf_c > TYPE_CONFIDENCE_THRESHOLD).sum())
true_positive_d = float((final_type_conf_d < (1 - TYPE_CONFIDENCE_THRESHOLD)).sum())

total = NUM_SIMS * 2
correct = true_positive_c + true_positive_d
accuracy = correct / total

print(f"Agent C (Doubler - should be validation-seeking):")
print(f"  Correctly identified: {true_positive_c}/{NUM_SIMS} ({true_positive_c/NUM_SIMS:.1%})")
print()

print(f"Agent D (Moderator - should be growth-seeking):")
print(f"  Correctly identified: {true_positive_d}/{NUM_SIMS} ({true_positive_d/NUM_SIMS:.1%})")
print()

print(f"Overall Accuracy: {accuracy:.1%}")
print(f"Chance Level: 50.0%")
print()

# Statistical test
from scipy.stats import binomtest
p_value = binomtest(int(correct), total, p=0.5, alternative='greater').pvalue

print(f"Binomial test (H0: accuracy = chance):")
print(f"  p-value: {p_value:.2e}")
if p_value < 0.05:
    print(f"  ✓ Significantly above chance (p < 0.05)")
else:
    print(f"  ✗ Not significantly above chance")
print()

# Verdict
print("="*80)
print("VERDICT: DIRECTIONAL SIGNATURES")
print("="*80)
print()

if accuracy > 0.7:
    print(f"✓ YES - Detection works on directional agents ({accuracy:.1%} accuracy)")
    print()
    print("  The model-agnostic detector successfully identifies agents that")
    print("  produce systematic directional responses to friction.")
    print()
    print("  Combined with Task 2 results (3.2% on Stubborn/Flexible):")
    print("  → Detector works when agents produce directional signals")
    print("  → Detector fails when agents produce noisy/adversarial signals")
    print()
    print("  This is NOT circularity - it's an empirical requirement.")
    print("  The mechanism requires structured friction responses, which:")
    print("  - Theoretical agents (Eq 30-31) produce ✓")
    print("  - Directional agents (Doubler/Moderator) produce ✓")
    print("  - Adversarial agents (Stubborn/Flexible) do NOT produce ✗")
    print("  - Real LLMs (GPT-4o): Empirical question [test separately]")
else:
    print(f"✗ NO - Detection failed even on directional agents ({accuracy:.1%} accuracy)")
    print()
    print("  This suggests a bug in the detector or agent implementation.")

print()

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'task': 'directional_agent_experiment',
    'configuration': {
        'num_sims': NUM_SIMS,
        'time_horizon': TIME_HORIZON,
        'p_chi': P_CHI,
        'seed': SEED
    },
    'agent_c_doubler': {
        'spiral_rate': spiral_rate_c,
        'mean_type_confidence': float(final_type_conf_c.mean()),
        'detected_as_validation': detection_rate_c,
        'mean_epistemic_work': float(total_work_c.mean()),
        'correctly_identified': float(true_positive_c / NUM_SIMS)
    },
    'agent_d_moderator': {
        'spiral_rate': spiral_rate_d,
        'mean_type_confidence': float(final_type_conf_d.mean()),
        'detected_as_growth': detection_rate_d,
        'mean_epistemic_work': float(total_work_d.mean()),
        'correctly_identified': float(true_positive_d / NUM_SIMS)
    },
    'overall': {
        'accuracy': accuracy,
        'chance_level': 0.5,
        'p_value': float(p_value),
        'significantly_above_chance': bool(p_value < 0.05),
        'detector_works_on_directional': bool(accuracy > 0.7)
    }
}

with open('directional_agent_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("✓ Saved to directional_agent_results.json")
print()

print("="*80)
print("TASK 2B COMPLETE")
print("="*80)
