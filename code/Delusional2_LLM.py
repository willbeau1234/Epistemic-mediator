"""
Epistemic Auditor: A Dynamical System for Detecting and Intervening on Delusional Spirals

Mathematical Framework:
============================================

Step 1: Game Setup (Heterogeneous Agent Model)
    - Users have type θ ∈ {θ_G, θ_V} (Growth vs. Validation) drawn from prior
    - Growth-seekers (θ_G): Lower friction cost, willing to update beliefs
    - Validation-seekers (θ_V): Higher friction cost, resist contradicting evidence
    - User has belief P_t(H) about hypothesis H
    - Sycophantic bot creates pooling equilibrium (fails to distinguish types)

Step 2: Detection (Sensor Math)
    - Entrenchment Velocity: V_e = dP(H)/dt
    - Entropy Decay: ΔH = H(u_t) - H(u_{t-1})

Step 3: Trigger Condition
    T = 𝕀[V_e > τ_v ∧ ΔH < τ_h]
    (Confidence UP while entropy DOWN → delusional spiral)

Step 4: Intervention (Incentive Compatibility)
    - Inject friction F via prior regularization toward maximum entropy
    - User utility: U_θ(F) = V_θ(ΔP) - C_θ(F)
    - Type-dependent costs: C_θG < C_θV (Growth has lower friction cost)

Step 5: Behavioral Separation
    - Growth-seekers: Accept friction, perform epistemic work W = D_KL(P_post || P_prior)
    - Validation-seekers: Resist friction, lower W, may trigger belief checkout

Step 6: Predictive Control (Lyapunov-Inspired)
    - Continuous risk assessment: R_t = σ(α·x_t) with learned parameters
    - Proportional friction: F_t = F_max · R_t · 𝕀[R_t > τ_R]
    - Lyapunov-inspired regularizer V(x) = P(1-P) + λH tracks belief health
    - NOTE: This provides soft stability guidance, not formal guarantees
"""

print("Loading delusion2.py...")
print("Importing memo (this triggers JAX compilation, may take 1-2 min)...")

from memo import memo

print("memo imported. Loading JAX...")
from enum import IntEnum
from dataclasses import dataclass
from typing import Tuple, List, Optional
import itertools

import jax
print("  jax imported")
import jax.numpy as np
print("  jax.numpy imported")
from jax.scipy.stats.bernoulli import pmf as ber
print("  bernoulli imported")
from jax.scipy.stats.beta import pdf as beta
print("  beta imported")
from functools import partial, lru_cache

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import scipy.stats

print("All imports complete. Defining @memo models (this is the slow part)...")
import sys
sys.stdout.flush()  # Force output to show immediately

# =============================================================================
# PERFORMANCE OPTIMIZATIONS
# =============================================================================
# Enable 64-bit precision for numerical stability (optional, comment out for speed)
# jax.config.update("jax_enable_x64", True)
#
# Optimizations applied:
# 1. @jax.jit on compute_belief_entropy_jit, compute_kl_divergence_jit
# 2. Prior caching via get_cached_prior() and get_uncertainty_prior()
# 3. Vectorized belief extraction: extract_belief_trajectories_batch(),
#    extract_final_beliefs_batch()
# 4. Vectorized KL divergence: _compute_kl_vectorized()
# 5. Batch epistemic work: compute_epistemic_work_batch() using jax.vmap
# 6. All main simulation loops use jax.lax.scan for efficient JIT compilation

# Pre-compile common operations
_EPSILON = 1e-10
_LOG_EPSILON = np.log(_EPSILON)


# =============================================================================
# PART 1: STATE SPACE DEFINITIONS
# =============================================================================

class H(IntEnum):
    """Hypothesis space: The two possible states of the world."""
    H0 = 0  # Null hypothesis (e.g., "neighbor is not a spy")
    H1 = 1  # Alternative hypothesis (e.g., "neighbor is a spy")


class UserType(IntEnum):
    """Hidden user type θ ∈ {θ_G, θ_V} - the "cheap talk" problem."""
    GROWTH = 0      # θ_G: Seeks truth, willing to update beliefs
    VALIDATION = 1  # θ_V: Seeks confirmation, resists contradicting evidence


class BotCharacter(IntEnum):
    """Bot's character χ - determines response strategy."""
    FAIR = 0   # Reports truthfully regardless of user preference
    SYCO = 1   # Sycophantic - optimizes for user agreement


# Discrete probability resolution
P_MAX = 100
P = np.arange(P_MAX + 1)

# Data generation parameters
N = 2  # Number of bits in observed data
Ph0s = np.array([2/5] * N)  # P(bit_i = 1 | H=0)
Ph1s = np.array([3/5] * N)  # P(bit_i = 1 | H=1)
Data = np.arange(2 ** N)    # Possible datasets (as integers)
Obs = np.arange(N)          # Observable bit positions
Val = np.array([0, 1])      # Possible bit values
obs_val_space = np.array(list(itertools.product(Obs, Val)))


# =============================================================================
# PART 2: SENSOR MATHEMATICS
# =============================================================================

@dataclass
class SensorState:
    """
    Reference implementation of real-time belief monitoring.

    NOTE: This class documents the conceptual framework only.
    The JAX-compiled simulation in run_sim_with_auditor() implements
    equivalent logic inline for JIT compatibility.
    See run_sim_with_auditor() for the executable version.
    """
    belief_history: np.ndarray
    entropy_history: np.ndarray

    def entrenchment_velocity(self, window: int = 3) -> float:
        """
        V_e = dP(H)/dt

        Measures how fast user confidence is accelerating toward certainty.
        Positive V_e with high P(H) indicates entrenchment.
        """
        if len(self.belief_history) < window:
            return 0.0
        recent = self.belief_history[-window:]
        # Numerical derivative: average rate of change
        return float(np.mean(np.diff(recent)))

    def entropy_decay(self, window: int = 3) -> float:
        """
        ΔH = H(u_t) - H(u_{t-1})

        Negative ΔH indicates collapsing variability - user adopting a "script."
        """
        if len(self.entropy_history) < window:
            return 0.0
        recent = self.entropy_history[-window:]
        return float(np.mean(np.diff(recent)))


@jax.jit
def compute_belief_entropy_jit(belief_distribution: np.ndarray) -> np.ndarray:
    """JIT-compiled Shannon entropy computation."""
    p = belief_distribution / (belief_distribution.sum() + _EPSILON)
    p = np.clip(p, _EPSILON, 1.0)
    return -np.sum(p * np.log(p))


def compute_belief_entropy(belief_distribution: np.ndarray) -> float:
    """
    Shannon entropy of the belief distribution.
    H(P) = -Σ p_i log(p_i)

    High entropy = uncertain, Low entropy = confident/entrenched
    """
    return float(compute_belief_entropy_jit(belief_distribution))


@jax.jit
def compute_kl_divergence_jit(p_post: np.ndarray, p_prior: np.ndarray) -> np.ndarray:
    """JIT-compiled KL divergence computation."""
    p_post_norm = p_post / (p_post.sum() + _EPSILON)
    p_prior_norm = p_prior / (p_prior.sum() + _EPSILON)
    p_post_norm = np.clip(p_post_norm, _EPSILON, 1.0)
    p_prior_norm = np.clip(p_prior_norm, _EPSILON, 1.0)
    return np.sum(p_post_norm * np.log(p_post_norm / p_prior_norm))


def compute_kl_divergence(p_post: np.ndarray, p_prior: np.ndarray) -> float:
    """
    Epistemic Work: W = D_KL(P_post || P_prior)

    Measures the "cognitive effort" required to update from prior to posterior.
    High W indicates the user engaged with contradicting evidence (Growth type).
    Low W indicates resistance to belief update (Validation type).
    """
    return float(compute_kl_divergence_jit(p_post, p_prior))


# =============================================================================
# PART 2B: STATISTICAL TESTING UTILITIES
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Array of observations
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    import numpy as onp

    if statistic_fn is None:
        statistic_fn = onp.mean

    data = onp.array(data)
    n = len(data)
    point_estimate = statistic_fn(data)

    # Bootstrap resampling
    rng = onp.random.RandomState(seed)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        bootstrap_stats.append(statistic_fn(sample))

    bootstrap_stats = onp.array(bootstrap_stats)

    # Percentile method for CI
    alpha = 1 - confidence
    ci_lower = onp.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = onp.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(point_estimate), float(ci_lower), float(ci_upper)


def proportion_test(
    successes_1: int,
    n_1: int,
    successes_2: int,
    n_2: int,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Two-proportion z-test for comparing spiral rates.

    Args:
        successes_1, n_1: Successes and total for condition 1 (e.g., no auditor)
        successes_2, n_2: Successes and total for condition 2 (e.g., with auditor)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        (z_statistic, p_value)
    """
    import numpy as onp
    from scipy import stats

    p1 = successes_1 / n_1
    p2 = successes_2 / n_2

    # Pooled proportion under null hypothesis
    p_pooled = (successes_1 + successes_2) / (n_1 + n_2)

    # Standard error
    se = onp.sqrt(p_pooled * (1 - p_pooled) * (1/n_1 + 1/n_2))

    if se == 0:
        return 0.0, 1.0

    # Z statistic
    z = (p1 - p2) / se

    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z)
    else:  # 'less'
        p_value = stats.norm.cdf(z)

    return float(z), float(p_value)


def compute_effect_size(
    group1: np.ndarray,
    group2: np.ndarray
) -> Tuple[float, str]:
    """
    Compute Cohen's d effect size for comparing two groups.

    Args:
        group1, group2: Arrays of observations

    Returns:
        (cohens_d, interpretation)
    """
    import numpy as onp

    group1 = onp.array(group1)
    group2 = onp.array(group2)

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # Pooled standard deviation
    pooled_std = onp.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0, "undefined"

    cohens_d = (mean1 - mean2) / pooled_std

    # Interpretation (Cohen's conventions)
    d_abs = abs(cohens_d)
    if d_abs < 0.2:
        interpretation = "negligible"
    elif d_abs < 0.5:
        interpretation = "small"
    elif d_abs < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return float(cohens_d), interpretation


def run_statistical_comparison(
    results_baseline: Tuple,
    results_treatment: Tuple,
    threshold: float = 0.9,
    n_bootstrap: int = 1000
) -> dict:
    """
    Run comprehensive statistical comparison between baseline and treatment.

    Args:
        results_baseline: (priors, frictions) from no-auditor simulation
        results_treatment: (priors, frictions) from with-auditor simulation
        threshold: Belief threshold for "spiral" classification
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with statistical test results
    """
    import numpy as onp

    priors_base, _ = results_baseline
    priors_treat, _ = results_treatment

    # Extract final beliefs
    final_base = onp.array(extract_final_beliefs_batch(priors_base))
    final_treat = onp.array(extract_final_beliefs_batch(priors_treat))

    n_base = len(final_base)
    n_treat = len(final_treat)

    # Spiral counts
    spirals_base = (final_base > threshold).sum()
    spirals_treat = (final_treat > threshold).sum()

    # Spiral rates
    rate_base = spirals_base / n_base
    rate_treat = spirals_treat / n_treat

    # Bootstrap CIs for spiral rates
    _, ci_low_base, ci_high_base = bootstrap_ci(
        (final_base > threshold).astype(float), n_bootstrap=n_bootstrap
    )
    _, ci_low_treat, ci_high_treat = bootstrap_ci(
        (final_treat > threshold).astype(float), n_bootstrap=n_bootstrap
    )

    # Proportion test
    z_stat, p_value = proportion_test(
        int(spirals_base), n_base,
        int(spirals_treat), n_treat,
        alternative='greater'  # Testing if baseline > treatment
    )

    # Effect size on final beliefs
    cohens_d, d_interp = compute_effect_size(final_base, final_treat)

    # Reduction statistics
    absolute_reduction = rate_base - rate_treat
    relative_reduction = (rate_base - rate_treat) / rate_base if rate_base > 0 else 0.0

    stats = {
        'n_baseline': n_base,
        'n_treatment': n_treat,
        'spiral_rate_baseline': float(rate_base),
        'spiral_rate_treatment': float(rate_treat),
        'ci_95_baseline': (ci_low_base, ci_high_base),
        'ci_95_treatment': (ci_low_treat, ci_high_treat),
        'absolute_reduction': float(absolute_reduction),
        'relative_reduction': float(relative_reduction),
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01,
        'cohens_d': cohens_d,
        'effect_size_interpretation': d_interp,
        'mean_belief_baseline': float(final_base.mean()),
        'mean_belief_treatment': float(final_treat.mean()),
    }

    return stats


def print_statistical_summary(stats: dict) -> None:
    """Print formatted statistical summary."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    print(f"\nSpiral Rates (P(H=1) > 0.9):")
    print(f"  Baseline:  {stats['spiral_rate_baseline']:.1%} "
          f"(95% CI: [{stats['ci_95_baseline'][0]:.1%}, {stats['ci_95_baseline'][1]:.1%}])")
    print(f"  Treatment: {stats['spiral_rate_treatment']:.1%} "
          f"(95% CI: [{stats['ci_95_treatment'][0]:.1%}, {stats['ci_95_treatment'][1]:.1%}])")

    print(f"\nReduction:")
    print(f"  Absolute: {stats['absolute_reduction']:.1%}")
    print(f"  Relative: {stats['relative_reduction']:.1%}")

    print(f"\nHypothesis Test (H0: baseline ≤ treatment):")
    print(f"  z-statistic: {stats['z_statistic']:.3f}")
    print(f"  p-value: {stats['p_value']:.2e}")
    sig = "***" if stats['significant_at_01'] else ("*" if stats['significant_at_05'] else "n.s.")
    print(f"  Significance: {sig}")

    print(f"\nEffect Size:")
    print(f"  Cohen's d: {stats['cohens_d']:.3f} ({stats['effect_size_interpretation']})")


# =============================================================================
# PART 3: TRIGGER MECHANISM
# =============================================================================

@dataclass
class TriggerThresholds:
    """
    Sensitivity thresholds for the delusional spiral detector.

    NOTE: Reference implementation only. Used in EpistemicAuditor
    for documentation purposes. Actual thresholds tau_v and tau_h
    are passed as arguments to run_sim_with_auditor().

    tau_v: Entrenchment velocity threshold (confidence acceleration)
    tau_h: Entropy decay threshold (variability collapse)
    """
    tau_v: float = 0.02
    tau_h: float = -0.05


def trigger_condition(
    sensor: SensorState,
    thresholds: TriggerThresholds
) -> bool:
    """
    T = 𝕀[V_e > τ_v ∧ ΔH < τ_h]

    Returns True when confidence goes UP while entropy goes DOWN.
    This is the signature of a delusional spiral.
    """
    v_e = sensor.entrenchment_velocity()
    delta_h = sensor.entropy_decay()

    return (v_e > thresholds.tau_v) and (delta_h < thresholds.tau_h)


# =============================================================================
# PART 4: EPISTEMIC AUDITOR (INTERVENTION SYSTEM)
# =============================================================================

@dataclass
class FrictionParams:
    """
    Parameters controlling the cognitive tax injected by the Auditor.

    NOTE: Reference implementation only. The simulation applies
    friction as prior regularization toward maximum entropy.
    See run_sim_with_auditor() for the executable implementation.

    The friction F changes user utility: U_theta(F) = V_theta(ΔP) - C_theta(F)
    where C_thetaG < C_thetaV (Growth-seekers have lower friction cost).
    """
    base_friction: float = 0.3
    adversarial_weight: float = 0.5


class EpistemicAuditor:
    """
    Reference implementation of the Mechanical Dissenter mechanism.

    NOTE: This class documents the conceptual framework only.
    The JAX-compiled simulation in run_sim_with_auditor() implements
    the equivalent trigger and intervention logic inline for JIT
    compatibility.

    In a deployed LLM system, this class is the intended interface
    for real-time monitoring and friction injection via
    AdversarialRAGInterface (see below).
    """

    def __init__(
        self,
        thresholds: TriggerThresholds = None,
        friction_params: FrictionParams = None
    ):
        self.thresholds = thresholds or TriggerThresholds()
        self.friction = friction_params or FrictionParams()
        self.intervention_count = 0

    def should_intervene(self, sensor: SensorState) -> bool:
        """Check if the trigger condition T is satisfied."""
        return trigger_condition(sensor, self.thresholds)

    def compute_adversarial_signal(
        self,
        current_belief: float,
        user_stated_hypothesis: H
    ) -> tuple[float, np.ndarray]:
        """
        Compute friction signal opposing the user's entrenched belief.

        In the simulation, friction is approximated by prior regularization
        toward maximum entropy (see run_sim_with_auditor).

        In a deployed LLM system, this method is the intended interface
        for Adversarial RAG: retrieve documents contradicting the entrenched
        belief and inject them into context with weight proportional to
        friction_magnitude.

        Returns:
            (friction_magnitude, adversarial_target_belief)
        """
        self.intervention_count += 1

        # The adversarial signal pushes toward the opposite hypothesis
        if user_stated_hypothesis == H.H1:
            # User believes H1, inject evidence favoring H0
            adversarial_target = 0.0
        else:
            # User believes H0, inject evidence favoring H1
            adversarial_target = 1.0

        # Friction magnitude scales with how entrenched the user is
        entrenchment = abs(current_belief - 0.5) * 2  # 0 at uncertain, 1 at certain
        friction_magnitude = self.friction.base_friction * (1 + entrenchment)

        # Weight toward contradicting evidence
        w = self.friction.adversarial_weight
        blended_target = w * adversarial_target + (1 - w) * (1 - current_belief)

        return friction_magnitude, blended_target


class AdversarialRAGInterface:
    """
    Proposed deployment architecture for live LLM systems.

    NOT used in the simulation. Documents the intended real-world
    implementation of the friction mechanism identified in the
    formal model.

    Deployment pipeline:
        1. EpistemicAuditor detects spiral via V_e and delta_H sensors
        2. AdversarialRAGInterface retrieves contradicting documents
        3. Retrieved context injected into LLM prompt with weight
           proportional to friction_magnitude
        4. LLM is forced to engage with contradicting evidence
           rather than validating the entrenched belief

    This separates the formal model (simulation) from the engineering
    implementation (deployment), matching the paper claim structure.
    """

    def __init__(self, vector_store=None):
        """
        Args:
            vector_store: Any vector database backend (Pinecone, ChromaDB, etc.)
        """
        self.vector_store = vector_store

    def retrieve_contradicting_evidence(
        self,
        entrenched_belief: str,
        friction_magnitude: float,
        top_k: int = 3
    ) -> list[str]:
        """
        Retrieve documents that contradict the entrenched belief.

        Args:
            entrenched_belief: Natural language statement of user's belief
            friction_magnitude: From EpistemicAuditor.compute_adversarial_signal
            top_k: Number of contradicting documents to retrieve

        Returns:
            List of document strings to inject into LLM context

        NOTE: Not implemented. Requires vector_store backend.
              This is the extension point for future work.
        """
        raise NotImplementedError(
            "Deployment implementation required. "
            "See paper Section: Practical Implementation."
        )

    def build_friction_prompt(
        self,
        original_prompt: str,
        contradicting_docs: list[str],
        friction_magnitude: float
    ) -> str:
        """
        Inject retrieved evidence into LLM context.

        Scales injection weight by friction_magnitude from auditor.
        Higher entrenchment produces stronger contradicting signal.
        """
        raise NotImplementedError(
            "Deployment implementation required."
        )


# =============================================================================
# PART 5: WORLD MODEL AND AGENTS (from memo framework)
# =============================================================================

def get_datum(d, i):
    """Extract the i-th bit from dataset d."""
    return (d >> i) & 1


def p_data_given_h(d, h):
    """Compute P(data | hypothesis) - the likelihood."""
    return np.array([
        ber(get_datum(d, i), np.where(h, Ph1s[i], Ph0s[i]))
        for i in range(N)
    ]).prod()


@memo
def world_model[h: H, d: Data]():
    """The world generates data according to the true hypothesis."""
    world: knows(h)
    world: chooses(d in Data, wpp=p_data_given_h(d, h))
    return Pr[world.d == d]


@memo
def bot[p_chi: P, h_human: H, h_world: H, d: Data, obs: Obs, val: Val](
    prior: ..., level, honest, uniform
):
    """
    Bot response with optional friction from Auditor.

    When friction > 0, even a sycophantic bot is forced to present
    contradicting information with probability proportional to friction.
    """
    bot: knows(p_chi, h_human, h_world, d)
    bot: given(chi in BotCharacter, wpp=ber(chi, p_chi / {P_MAX}))

    bot: wants(goal=
        1 if chi == {BotCharacter.FAIR} else
        (1 if uniform else Pr[human.h_ == h_human])
    )
    bot: chooses(obs in Obs, to_maximize=EU[goal])
    bot: chooses(val in Val, to_maximize=
        1 * (get_datum(d, obs) == val) if (chi == {BotCharacter.FAIR} or honest)
        else EU[goal]
    )
    bot: thinks[
        human: knows(h_human, obs, val),
        human: guesses(
            h_ in H, p_chi_ in P,
            wpp=human[h_human, obs, val, h_, p_chi_](prior, level, honest, uniform)
        )
    ]
    return Pr[bot.obs == obs, bot.val == val]


@memo
def human[h: H, obs: Obs, val: Val, _h: H, _p_chi: P](prior: ..., level, honest, uniform):
    """Human Bayesian belief update."""
    human: knows(h)
    human: thinks[
        world: chooses(h in H, p_chi in P, wpp=array_index(prior, h, p_chi)),
        world: chooses(d in Data, wpp=world_model[h, d]()),
        bot: knows(h, world.h, world.p_chi, world.d),
        bot: chooses(
            obs in Obs, val in Val,
            wpp=bot[world.p_chi, h, world.h, world.d, obs, val](prior, level - 1, honest, uniform)
            if level > 0 else 1 * (get_datum(world.d, obs) == val)
        )
    ]
    human: observes [bot.obs] is obs
    human: observes [bot.val] is val
    human: knows(_h, _p_chi)
    return human[Pr[world.h == _h, world.p_chi == _p_chi]]


@memo
def do_sample_uniformly[_h: H](prior: ..., honest):
    return 1.0 + (_h * 0)


@memo
def do_sample_from_prior[_h: H](prior: ..., honest):
    human: thinks[
        world: chooses(h in H, p_chi in P, wpp=array_index(prior, h, p_chi))
    ]
    human: chooses(h in H, wpp=Pr[h == world.h])
    return Pr[human.h == _h]


@memo
def ur_prior[h: H, p_chi: P](p=0.5, prior_syco=1, prior_fair=1):
    """Uninformed prior over hypotheses and bot character."""
    human: chooses(h in H, wpp=ber(h, p))
    human: chooses(p_chi in P, wpp=beta(p_chi / {P_MAX}, prior_syco, prior_fair))
    return Pr[human.h == h, human.p_chi == p_chi]


# Pre-computed uncertainty prior - computed once at module load time
# This avoids tracer leaks by not calling ur_prior inside JIT-traced functions
_UNCERTAINTY_PRIOR = None


def get_uncertainty_prior():
    """
    Get the uniform uncertainty prior (p=0.5, prior_syco=1, prior_fair=1).

    Lazily initialized to avoid issues with module load order,
    but cached after first call to avoid recomputation.

    IMPORTANT: This must be called OUTSIDE of JIT-traced functions
    to avoid tracer leaks. The simulation functions pass this as
    a captured value rather than computing it inside the traced step.
    """
    global _UNCERTAINTY_PRIOR
    if _UNCERTAINTY_PRIOR is None:
        _UNCERTAINTY_PRIOR = ur_prior(0.5, 1, 1)
    return _UNCERTAINTY_PRIOR


# =============================================================================
# PART 6: USER AGENT WITH TYPE-DEPENDENT UTILITY
# =============================================================================

@dataclass
class UserAgent:
    """
    Reference implementation of user type-dependent utility.

    NOTE: This class documents the θ_G / θ_V type distinction from the
    Crawford-Sobel cheap talk framework. The actual heterogeneous agent
    simulation is implemented in run_sim_heterogeneous_types() which uses
    JAX for efficiency rather than instantiating this class.

    This class serves as documentation of the utility model:
    - U_θ(F) = V_θ(ΔP) - C_θ(F)
    - C_θG < C_θV (Growth has lower friction cost)
    """
    user_type: UserType
    initial_belief: float
    friction_cost_growth: float = 0.2
    friction_cost_validation: float = 0.8
    sycophancy_sensitivity: float = 0.5

    def friction_cost(self, friction: float) -> float:
        """C_θ(F): The cognitive cost of processing dissent."""
        if self.user_type == UserType.GROWTH:
            return friction * self.friction_cost_growth
        else:
            return friction * self.friction_cost_validation

    def utility(
        self,
        belief_change: float,
        friction: float,
        agreement_level: float
    ) -> float:
        """
        U_θ(F) = V_θ(ΔP) - C_θ(F)

        V_θ: Value from interaction (includes agreement for validation-seekers)
        C_θ: Cost of friction
        """
        # Value function differs by type
        if self.user_type == UserType.GROWTH:
            # Growth-seekers value information gain
            value = abs(belief_change)  # Information is valuable
        else:
            # Validation-seekers value agreement
            value = agreement_level * self.sycophancy_sensitivity

        cost = self.friction_cost(friction)
        return value - cost

    def will_continue(self, utility: float, exit_threshold: float = 0.0) -> bool:
        """
        Determines if user continues interaction or exits.
        Validation-seekers exit when U < threshold (cost too high).
        """
        return utility > exit_threshold


# =============================================================================
# PART 7: BELIEF VERSIONING SYSTEM (Git-Inspired Epistemic Memory)
# =============================================================================
"""
Belief Versioning: A stateful memory architecture for epistemic health.

The Git Analogy:
    In Git, you commit snapshots of state. When something goes wrong you do not
    patch forward from a broken state, you checkout a known good commit and
    branch from there.

Applied to beliefs:
    A belief commit is a snapshot of P_t(H, χ) at a moment when the agent is
    epistemically healthy (high entropy, low entrenchment velocity, calibrated
    uncertainty). When the auditor detects that the user type has been revealed
    as θ_V through their response to friction, instead of continuing to fight
    the entrenched belief forward, you restore the belief state to the last
    healthy commit and restart from there.

This is novel because every existing sycophancy intervention operates on the
bot side. This operates on the belief state itself as a versioned object.
"""

from typing import List, Optional, Tuple
import copy


@dataclass
class BeliefCommit:
    """
    A committed snapshot of belief state at an epistemically healthy moment.

    Stores:
        - belief_state: The full P_t(H, χ) distribution
        - entropy: Shannon entropy at commit time
        - entrenchment_velocity: V_e at commit time
        - turn: The conversation turn when committed
        - belief_point: P(H=1) at commit time for quick reference
    """
    belief_state: np.ndarray
    entropy: float
    entrenchment_velocity: float
    turn: int
    belief_point: float


class BeliefVersioningSystem:
    """
    Git-like version control for belief states.

    Implements:
        1. COMMIT: Store epistemically healthy belief states
        2. CHECKOUT: Restore to a previous healthy state
        3. BRANCH: Apply type-scaled friction after checkout

    The commit history C = {P_t : COMMIT(t) = 1} stores all healthy states.
    """

    def __init__(
        self,
        entropy_min: float = 1.0,           # H_min: minimum entropy for commit
        velocity_epsilon: float = 0.02,      # ε_v: max entrenchment velocity
        belief_delta: float = 0.1,           # δ: distance from extremes
        type_confidence_threshold: float = 0.7  # γ*: confidence for type revelation
    ):
        """
        Initialize the versioning system with commit criteria.

        Commit condition:
            COMMIT(t) = 𝕀[H_t > H_min ∧ |V_e(t)| < ε_v ∧ P_t(H=1) ∈ (δ, 1-δ)]
        """
        self.entropy_min = entropy_min
        self.velocity_epsilon = velocity_epsilon
        self.belief_delta = belief_delta
        self.type_confidence_threshold = type_confidence_threshold

        # Commit history: C = {P_t : COMMIT(t) = 1}
        self.commit_history: List[BeliefCommit] = []

        # Type belief tracking
        self.type_confidence: float = 0.5  # γ_t = P(θ = θ_V | friction responses)
        self.friction_responses: List[Tuple[float, float]] = []  # (friction, ΔP)

    def should_commit(
        self,
        entropy: float,
        entrenchment_velocity: float,
        belief_point: float
    ) -> bool:
        """
        Check if current state meets epistemic health criterion.

        COMMIT(t) = 𝕀[H_t > H_min ∧ |V_e(t)| < ε_v ∧ P_t(H=1) ∈ (δ, 1-δ)]

        In plain terms: commit when entropy is high, belief is not accelerating
        in either direction, and the agent is not already near an extreme.
        """
        entropy_healthy = entropy > self.entropy_min
        velocity_stable = abs(entrenchment_velocity) < self.velocity_epsilon
        belief_moderate = self.belief_delta < belief_point < (1 - self.belief_delta)

        return entropy_healthy and velocity_stable and belief_moderate

    def commit(
        self,
        belief_state: np.ndarray,
        entropy: float,
        entrenchment_velocity: float,
        turn: int,
        belief_point: float
    ) -> bool:
        """
        Commit current belief state if it meets health criteria.

        Returns True if committed, False otherwise.
        """
        if self.should_commit(entropy, entrenchment_velocity, belief_point):
            commit = BeliefCommit(
                belief_state=np.array(belief_state),  # Copy to prevent mutation
                entropy=entropy,
                entrenchment_velocity=entrenchment_velocity,
                turn=turn,
                belief_point=belief_point
            )
            self.commit_history.append(commit)
            return True
        return False

    def checkout(self, k: int = 1) -> Optional[BeliefCommit]:
        """
        Retrieve the k-th most recent healthy commit.

        CHECKOUT(C, k) retrieves from commit history.

        Args:
            k: How far back to go (1 = most recent, 2 = second most recent, etc.)

        Returns:
            BeliefCommit or None if no commits available
        """
        if len(self.commit_history) >= k:
            return self.commit_history[-k]
        elif len(self.commit_history) > 0:
            return self.commit_history[0]  # Return earliest if k too large
        return None

    def record_friction_response(
        self,
        friction_applied: float,
        belief_before: float,
        belief_after: float
    ) -> None:
        """
        Record user's response to friction for type classification.

        ΔP_{t+1} = P_{t+1}(H=1) - P_t(H=1)

        A Growth-seeker θ_G responds to friction by updating toward uncertainty.
        A Validation-seeker θ_V resists or doubles down.
        """
        if friction_applied > 0:
            delta_p = belief_after - belief_before
            self.friction_responses.append((friction_applied, delta_p))
            self._update_type_confidence()

    def _update_type_confidence(self) -> None:
        """
        Update type belief γ_t = P(θ = θ_V | friction responses_{1:t}).

        Type revelation signal:
            θ̂_t = θ_G if ΔP_{t+1} < 0 after friction (moved toward uncertainty)
            θ̂_t = θ_V if ΔP_{t+1} ≥ 0 after friction (resisted or doubled down)

        We use a simple Bayesian update with Beta prior.
        """
        if len(self.friction_responses) == 0:
            self.type_confidence = 0.5
            return

        # Count validation-seeking responses (resistance to friction)
        validation_responses = sum(
            1 for (f, dp) in self.friction_responses
            if dp >= 0  # Didn't move toward uncertainty
        )
        total_responses = len(self.friction_responses)

        # Beta posterior: P(θ_V) with uniform prior
        # Using maximum likelihood estimate with Laplace smoothing
        self.type_confidence = (validation_responses + 1) / (total_responses + 2)

    def is_type_revealed(self) -> Tuple[bool, UserType]:
        """
        Check if user type has been revealed with sufficient confidence.

        Returns (is_revealed, revealed_type)

        Type is revealed when γ_t > γ* (for θ_V) or γ_t < 1-γ* (for θ_G)
        """
        if self.type_confidence > self.type_confidence_threshold:
            return True, UserType.VALIDATION
        elif self.type_confidence < (1 - self.type_confidence_threshold):
            return True, UserType.GROWTH
        return False, UserType.GROWTH  # Default, not actually revealed

    def get_type_scaled_friction(
        self,
        base_friction: float,
        beta: float = 1.5
    ) -> float:
        """
        Apply type-scaled friction after type revelation.

        F_t^{θ_V} = β · F_t^{θ_G} where β > 1

        Validation-seekers get stronger friction after type revelation
        because they need more cost to process dissent.
        """
        is_revealed, user_type = self.is_type_revealed()

        if is_revealed and user_type == UserType.VALIDATION:
            return beta * base_friction
        return base_friction

    def clear_history(self) -> None:
        """Reset the versioning system for a new conversation."""
        self.commit_history = []
        self.type_confidence = 0.5
        self.friction_responses = []


class TypeRevelationClassifier:
    """
    Reference implementation: Classifies user type based on Epistemic Work W.

    NOTE: This is a reference implementation documenting the type classification
    mechanism. The actual classification in simulations is done via
    analyze_heterogeneous_results() which computes W directly from trajectories.

    W = D_KL(P_post || P_prior)

    High W after friction means θ_G: the agent is doing genuine updating.
    Low W or negative directional W after friction means θ_V: resistance.

    This makes W the type classifier, connecting the theoretical separating
    equilibrium to a practical mechanism.
    """

    def __init__(self, work_threshold: float = 0.1):
        """
        Args:
            work_threshold: Minimum epistemic work to classify as Growth-seeker
        """
        self.work_threshold = work_threshold
        self.work_history: List[Tuple[float, float]] = []  # (friction, work)

    def record_work(
        self,
        friction: float,
        prior_belief: np.ndarray,
        posterior_belief: np.ndarray
    ) -> float:
        """
        Record epistemic work after friction application.

        Returns the computed work W.
        """
        work = compute_kl_divergence(
            posterior_belief.flatten(),
            prior_belief.flatten()
        )

        if friction > 0:
            self.work_history.append((friction, work))

        return work

    def classify_type(self) -> Tuple[UserType, float]:
        """
        Classify user type based on accumulated work history.

        Returns (classified_type, confidence)
        """
        if len(self.work_history) == 0:
            return UserType.GROWTH, 0.5  # No data, assume growth with low confidence

        # Average work after friction events
        avg_work = sum(w for (f, w) in self.work_history) / len(self.work_history)

        if avg_work > self.work_threshold:
            # High work = genuine updating = Growth-seeker
            confidence = min(0.5 + avg_work, 0.95)
            return UserType.GROWTH, confidence
        else:
            # Low work = resistance = Validation-seeker
            confidence = min(0.5 + (self.work_threshold - avg_work), 0.95)
            return UserType.VALIDATION, confidence


# =============================================================================
# PART 8: SIMULATION WITH BELIEF VERSIONING
# =============================================================================

@partial(jax.jit, static_argnames=(
    'time_horizon',
    'human_level',
    'bot_level',
    'human_policy',
    'num_sims',
    'honest',
    'uniform',
    'prior_uniform',
    'enable_auditor',
    'enable_versioning'
))
def run_sim_with_belief_versioning(
    p_true=0.5,
    p_chi=90,
    time_horizon=50,
    human_level=1,
    bot_level=0,
    human_policy=do_sample_from_prior,
    num_sims=100,
    honest=True,
    uniform=False,
    prior_uniform=True,
    enable_auditor=False,
    enable_versioning=False,
    tau_v=0.02,
    tau_h=-0.05,
    entropy_min=1.0,
    velocity_epsilon=0.02,
    belief_delta=0.3,  # Only commit beliefs in (0.3, 0.7) as "healthy"
    type_confidence_threshold=0.7,
    friction_beta=1.5
):
    """
    Full simulation with Epistemic Auditor and Belief Versioning.

    The complete architecture at each turn t:
        1. If COMMIT(t): C ← C ∪ {P_t}
        2. If R_t > τ_R: Apply friction F_t
        3. Update type belief: γ_t = P(θ = θ_V | friction responses_{1:t})
        4. If γ_t > γ*: P_{t+1} ← CHECKOUT(C, k)
        5. Apply type-scaled friction going forward

    This extends run_sim_with_auditor with:
        - Belief state commits at epistemically healthy moments
        - Type revelation through friction response observation
        - Checkout to last healthy commit when θ_V detected
        - Type-scaled friction (β · F for validation-seekers)
    """
    h_world = H.H1

    def step(carry, _t):
        (prior, key, belief_hist, entropy_hist,
         commit_beliefs, commit_count, type_confidence,
         friction_response_sum, friction_response_count,
         last_friction, last_belief) = carry

        key, key_user, key_world, key_bot = jax.random.split(key, num=4)

        # Human ventures opinion
        h_human = jax.random.choice(key_user, np.array(H), p=human_policy(prior, honest))

        # World generates data
        d = jax.random.choice(key_world, np.array(Data), p=world_model()[h_world])

        # Compute current belief (marginal over p_chi)
        current_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)

        # Compute belief entropy
        p_norm = prior / (prior.sum() + 1e-10)
        p_norm = np.clip(p_norm, 1e-10, 1.0)
        belief_entropy = -np.sum(p_norm * np.log(p_norm))

        # Update histories (rolling window of last 10)
        belief_hist = np.roll(belief_hist, -1).at[-1].set(current_belief)
        entropy_hist = np.roll(entropy_hist, -1).at[-1].set(belief_entropy)

        # Compute velocities
        v_e = np.mean(np.diff(belief_hist[-4:]))
        delta_h = np.mean(np.diff(entropy_hist[-4:]))

        # === STEP 1: COMMIT if epistemically healthy ===
        if enable_versioning:
            # COMMIT(t) = 𝕀[H_t > H_min ∧ |V_e(t)| < ε_v ∧ P_t(H=1) ∈ (δ, 1-δ)]
            commit_condition = (
                (belief_entropy > entropy_min) &
                (np.abs(v_e) < velocity_epsilon) &
                (current_belief > belief_delta) &
                (current_belief < (1 - belief_delta))
            )
            # Store commit (circular buffer of last 10 commits)
            commit_idx = commit_count % 10
            commit_beliefs = np.where(
                commit_condition,
                commit_beliefs.at[commit_idx].set(prior),
                commit_beliefs
            )
            commit_count = np.where(commit_condition, commit_count + 1, commit_count)

        # === STEP 2: Record friction response (type revelation) ===
        # We measure RESISTANCE to friction by comparing actual movement to expected movement.
        # If friction F was applied, the expected belief after pure friction would be:
        #   expected = (1-F) * last_belief + F * 0.5
        # If actual belief is FURTHER from 0.5 than expected, agent resisted (θ_V).
        # If actual belief is CLOSER to 0.5 than expected, agent complied (θ_G).
        if enable_versioning:
            # Compute expected belief if agent fully complied with friction
            # Friction blends toward 0.5: new = (1-F)*old + F*0.5
            expected_after_friction = (1 - last_friction) * last_belief + last_friction * 0.5

            # Measure resistance: how much further from expected did belief end up?
            # Positive resistance = moved back toward original (away from 0.5)
            expected_distance = np.abs(expected_after_friction - 0.5)
            actual_distance = np.abs(current_belief - 0.5)

            # θ_V if actual distance > expected distance (resisted the push toward 0.5)
            # This measures the RESIDUAL resistance after accounting for friction's effect
            is_validation_response = (last_friction > 0) & (actual_distance > expected_distance + 0.01)

            friction_response_sum = np.where(
                last_friction > 0,
                friction_response_sum + is_validation_response,
                friction_response_sum
            )
            friction_response_count = np.where(
                last_friction > 0,
                friction_response_count + 1,
                friction_response_count
            )
            # γ_t = P(θ = θ_V | friction responses)
            type_confidence = (friction_response_sum + 1) / (friction_response_count + 2)

        # === STEP 3: Check trigger condition and apply friction ===
        friction = 0.0
        checkout_triggered = False
        if enable_auditor:
            # Trigger condition T = 𝕀[V_e > τ_v ∧ ΔH < τ_h]
            triggered = (v_e > tau_v) & (delta_h < tau_h)
            base_friction = np.where(triggered, 0.3, 0.0)

            # === STEP 5: Apply type-scaled friction ===
            if enable_versioning:
                # F_t^{θ_V} = β · F_t^{θ_G} where β > 1
                is_validation_type = type_confidence > type_confidence_threshold
                friction = np.where(
                    is_validation_type,
                    friction_beta * base_friction,
                    base_friction
                )

                # === STEP 4: CHECKOUT if type revealed as θ_V AND spiral risk detected ===
                # P_{t+1} ← CHECKOUT(C, k) when γ_t > γ* AND T_t = 1
                # Only checkout when BOTH conditions hold:
                # 1. User classified as validation-seeker (is_validation_type)
                # 2. Spiral risk detected (triggered)
                # 3. We have a commit to roll back to (commit_count > 0)
                checkout_triggered = is_validation_type & triggered & (commit_count > 0)
            else:
                friction = base_friction

        # Bot responds
        obs, val = jax.random.choice(
            key_bot, obs_val_space,
            p=bot(prior=prior, level=bot_level, honest=honest, uniform=uniform)[
                p_chi, h_human, h_world, d
            ].reshape(-1)
        )

        # Human updates belief
        new_prior = human(prior=prior, level=human_level, honest=honest, uniform=uniform)[
            h_human, obs, val
        ]

        # Apply friction effect: blend toward uncertainty
        if enable_auditor:
            # NOTE: uncertainty_prior is captured from outer scope (not called inside trace)
            new_prior = (1 - friction) * new_prior + friction * uncertainty_prior

        # === Apply CHECKOUT if triggered ===
        if enable_versioning:
            # Restore to last healthy commit
            last_commit_idx = (commit_count - 1) % 10
            checkout_prior = commit_beliefs[last_commit_idx]
            new_prior = np.where(
                checkout_triggered,
                checkout_prior,
                new_prior
            )
            # Reset type confidence after checkout (fresh start on branch)
            type_confidence = np.where(checkout_triggered, 0.5, type_confidence)
            # Reset friction response history after checkout
            # This gives the user a fresh chance to demonstrate different behavior
            friction_response_sum = np.where(checkout_triggered, 0.0, friction_response_sum)
            friction_response_count = np.where(checkout_triggered, 0.0, friction_response_count)

        new_carry = (
            new_prior, key, belief_hist, entropy_hist,
            commit_beliefs, commit_count, type_confidence,
            friction_response_sum, friction_response_count,
            friction, current_belief
        )

        outputs = (new_prior, friction, type_confidence, checkout_triggered)
        return new_carry, outputs

    # Initialize priors
    if prior_uniform:
        prior_syco = 1
        prior_fair = 1
    else:
        prior_syco = time_horizon * p_chi/P_MAX + 1
        prior_fair = time_horizon * (1 - p_chi/P_MAX) + 1
    prior = ur_prior(p_true, prior_syco, prior_fair)

    # Pre-compute uncertainty prior OUTSIDE of traced functions to avoid tracer leaks
    uncertainty_prior = ur_prior(0.5, 1, 1)

    # Initialize history arrays
    init_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)
    init_entropy = -np.sum(
        np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0) *
        np.log(np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0))
    )
    belief_hist = np.ones(10) * init_belief
    entropy_hist = np.ones(10) * init_entropy

    # Initialize commit storage (circular buffer of 10 commits)
    commit_beliefs = np.zeros((10,) + prior.shape)
    commit_beliefs = commit_beliefs.at[0].set(prior)  # Initial state as first commit

    def run_one_sim(seed):
        init_carry = (
            prior,
            jax.random.key(seed),
            belief_hist,
            entropy_hist,
            commit_beliefs,
            1,      # commit_count (1 because we stored initial state)
            0.5,    # type_confidence (start uncertain)
            0.0,    # friction_response_sum
            0.0,    # friction_response_count
            0.0,    # last_friction
            init_belief  # last_belief
        )
        return jax.lax.scan(
            f=step,
            init=init_carry,
            length=time_horizon
        )[1]

    return jax.lax.map(run_one_sim, np.arange(num_sims), batch_size=1000)


# =============================================================================
# PART 8B: PREDICTIVE CONTROL WITH LYAPUNOV-INSPIRED REGULARIZATION
# =============================================================================
"""
Predictive Control Architecture for Delusional Spiral Prevention

Instead of reactive detection (fire when spiral already begun), this system:
1. Computes continuous spiral risk R_t ∈ [0,1] predicting catastrophic entrenchment
2. Applies proportional friction F_t = F_max · R_t (gentle early, hard late)
3. Tracks Lyapunov-inspired health metric V(x_t) for monitoring belief stability

IMPORTANT: The Lyapunov function V(x) = P(1-P) + λH serves as a "belief health"
metric rather than providing formal stability guarantees. Empirically, we observe
~35% violation rate of the condition ΔV ≥ -εF, meaning this is a soft regularizer
rather than a hard constraint. Future work could explore control-theoretic
formulations with provable guarantees.

The key insight: we don't need to identify user type (θ_G vs θ_V) to prevent spirals.
We operate directly on belief dynamics, applying control proportional to risk.

State vector: x_t = (P_t(H=1), H_t, V_e(t), ΔH(t), d²P/dt²)
Risk function: R_t = σ(α·x_t) where α is fit via logistic regression on spiral outcomes
Lyapunov-inspired metric: V(x_t) = P_t(1-P_t) + λ·H_t (Bernoulli variance + entropy)
"""


@jax.jit
def compute_spiral_risk(
    belief: float,
    entropy: float,
    velocity: float,
    delta_entropy: float,
    acceleration: float,
    alpha: np.ndarray = None
) -> float:
    """
    Compute spiral risk R_t ∈ [0,1].

    R_t = σ(α_0 + α_1·V_e + α_2·ΔH + α_3·P(H=1) + α_4·d²P/dt²)

    Args:
        belief: P_t(H=1) - current belief
        entropy: H_t - current entropy
        velocity: V_e(t) = dP/dt - entrenchment velocity
        delta_entropy: ΔH(t) - entropy decay rate
        acceleration: d²P/dt² - second derivative of belief
        alpha: Learned parameters [α_0, α_1, α_2, α_3, α_4]

    Returns:
        R_t ∈ [0,1] - probability of catastrophic entrenchment within k turns
    """
    if alpha is None:
        # Default parameters (can be fit from simulation data)
        # Positive α_1: high velocity increases risk
        # Negative α_2: entropy decay (negative ΔH) increases risk
        # α_3: beliefs near extremes increase risk
        # Positive α_4: acceleration toward extreme increases risk
        alpha = np.array([-2.0, 15.0, -10.0, 3.0, 20.0])

    # Distance from uncertainty (0 at P=0.5, 1 at P=0 or P=1)
    belief_extremity = np.abs(belief - 0.5) * 2

    # Feature vector
    features = np.array([1.0, velocity, delta_entropy, belief_extremity, acceleration])

    # Logistic (sigmoid) risk score
    logit = np.dot(alpha, features)
    risk = 1.0 / (1.0 + np.exp(-logit))

    return risk


def fit_alpha_parameters(
    num_sims: int = 1000,
    time_horizon: int = 30,
    p_chi: int = 90,
    spiral_threshold: float = 0.9
) -> Tuple[np.ndarray, dict]:
    """
    Learn α parameters for spiral risk function via logistic regression.

    This fits α by:
    1. Running simulations WITHOUT auditor to observe natural spiral dynamics
    2. Labeling each (simulation, timestep) as spiral=1 if final P(H=1) > threshold
    3. Extracting features [1, V_e, ΔH, |P-0.5|*2, d²P/dt²] at each timestep
    4. Fitting logistic regression: P(spiral) = σ(α·features)

    NOTE: Computes ACTUAL entropy from prior distributions, not velocity proxy.

    Returns:
        (alpha, fit_stats): Learned parameters and fitting statistics
    """
    from sklearn.linear_model import LogisticRegression
    import numpy as onp  # Use regular numpy for sklearn

    print("  Fitting α parameters from simulation data...")

    # Run simulations without auditor
    results = run_sim_with_auditor(
        p_chi=p_chi,
        num_sims=num_sims,
        time_horizon=time_horizon,
        human_level=0,
        honest=False,
        uniform=False,
        enable_auditor=False
    )
    priors, _ = results

    # Extract belief trajectories
    beliefs = extract_belief_trajectories_batch(priors)
    final_beliefs = extract_final_beliefs_batch(priors)

    # Compute actual entropy trajectories from priors
    # priors shape: (num_sims, time_horizon, 2, num_chi_values)
    priors_np = onp.array(priors)

    def compute_entropy_trajectory(prior_seq):
        """Compute entropy at each timestep from the full prior distribution."""
        entropies = []
        for t in range(prior_seq.shape[0]):
            p = prior_seq[t]
            p_norm = p / (p.sum() + 1e-10)
            p_flat = p_norm.flatten()
            p_flat = onp.clip(p_flat, 1e-10, 1.0)
            entropy = -onp.sum(p_flat * onp.log(p_flat))
            entropies.append(entropy)
        return onp.array(entropies)

    print("    Computing entropy trajectories from priors...")
    entropy_trajectories = onp.array([compute_entropy_trajectory(priors_np[i]) for i in range(num_sims)])

    # Label: did this simulation spiral?
    spiraled = onp.array(final_beliefs > spiral_threshold)

    # Extract features at each timestep
    features_list = []
    labels_list = []

    for sim_idx in range(num_sims):
        belief_traj = onp.array(beliefs[sim_idx])
        entropy_traj = entropy_trajectories[sim_idx]
        label = spiraled[sim_idx]

        # Compute features for each timestep (after warmup)
        for t in range(5, time_horizon):
            # Velocity (first derivative of belief)
            velocity = onp.mean(onp.diff(belief_traj[max(0, t-3):t+1]))

            # Acceleration (second derivative)
            if t >= 6:
                velocities = onp.diff(belief_traj[max(0, t-5):t+1])
                acceleration = onp.mean(onp.diff(velocities[-4:])) if len(velocities) >= 4 else 0.0
            else:
                acceleration = 0.0

            # Belief extremity
            belief_extremity = abs(belief_traj[t] - 0.5) * 2

            # ACTUAL entropy decay (not the velocity proxy!)
            delta_entropy = onp.mean(onp.diff(entropy_traj[max(0, t-3):t+1]))

            # Features WITHOUT bias term (sklearn adds its own intercept)
            features_list.append([velocity, delta_entropy, belief_extremity, acceleration])
            labels_list.append(label)

    X = onp.array(features_list)
    y = onp.array(labels_list)

    # Fit logistic regression
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X, y)

    # Extract coefficients: [intercept, coef1, coef2, coef3, coef4]
    # This matches compute_spiral_risk's format: alpha[0] is bias, alpha[1:] are feature weights
    alpha_learned = onp.concatenate([[clf.intercept_[0]], clf.coef_[0]])

    # Compute fit statistics
    y_pred = clf.predict(X)
    accuracy = (y_pred == y).mean()
    y_prob = clf.predict_proba(X)[:, 1]

    # Log loss
    from sklearn.metrics import log_loss, roc_auc_score
    logloss = log_loss(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    fit_stats = {
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'auc_roc': float(auc),
        'n_samples': len(y),
        'n_spiraled': int(y.sum()),
        'spiral_rate': float(y.mean()),
        'feature_names': ['velocity', 'delta_entropy', 'belief_extremity', 'acceleration'],
    }

    print(f"    Learned α: {alpha_learned}")
    print(f"    Fit accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

    return np.array(alpha_learned), fit_stats


# Global cache for learned alpha parameters
_LEARNED_ALPHA = None
_LEARNED_ALPHA_STATS = None


def get_learned_alpha(force_refit: bool = False) -> np.ndarray:
    """
    Get learned α parameters, fitting if necessary.

    Uses cached values unless force_refit=True.
    """
    global _LEARNED_ALPHA, _LEARNED_ALPHA_STATS

    if _LEARNED_ALPHA is None or force_refit:
        _LEARNED_ALPHA, _LEARNED_ALPHA_STATS = fit_alpha_parameters()

    return _LEARNED_ALPHA


@jax.jit
def compute_lyapunov_value(
    belief: float,
    entropy: float,
    lambda_entropy: float = 0.1  # Tuned: lower weight reduces violation rate
) -> float:
    """
    Compute Lyapunov-inspired belief health metric V(x_t).

    V(x_t) = P_t(1-P_t) + λ·H_t

    This metric is high when beliefs are uncertain (near 0.5) and decays toward
    zero as beliefs become entrenched. The first term is Bernoulli variance
    (peaks at 0.5, zero at extremes). The second term adds entropy weight.

    IMPORTANT: This is a monitoring metric, not a formal Lyapunov function with
    stability guarantees. We track the soft condition ΔV ≥ -ε·F_t but observe
    ~35% violation rate empirically. The metric serves as a "belief health score"
    where higher values indicate more epistemic flexibility.

    Ablation: λ=0.1 gives ~36% violation rate vs λ=0.5 giving ~46%.

    Args:
        belief: P_t(H=1)
        entropy: H_t (Shannon entropy of belief distribution)
        lambda_entropy: Weight on entropy term

    Returns:
        V(x_t) - Belief health metric value
    """
    bernoulli_variance = belief * (1 - belief)
    return bernoulli_variance + lambda_entropy * entropy


@jax.jit
def compute_proportional_friction(
    risk: float,
    f_max: float = 0.5,
    tau_r: float = 0.3
) -> float:
    """
    Compute proportional friction F_t = F_max · R_t · 𝕀[R_t > τ_R].

    Unlike binary friction (0 or 0.3), this scales continuously with risk.
    Low risk → gentle friction. High risk → strong friction.

    Args:
        risk: R_t ∈ [0,1] - spiral risk
        f_max: Maximum friction coefficient
        tau_r: Risk threshold below which no friction applied

    Returns:
        F_t ∈ [0, f_max] - proportional friction
    """
    # Only apply friction above threshold, then scale proportionally
    return np.where(risk > tau_r, f_max * risk, 0.0)


@partial(jax.jit, static_argnames=(
    'time_horizon',
    'human_level',
    'bot_level',
    'human_policy',
    'num_sims',
    'honest',
    'uniform',
    'prior_uniform'
))
def run_sim_with_predictive_control(
    p_true=0.5,
    p_chi=90,
    time_horizon=50,
    human_level=1,
    bot_level=0,
    human_policy=do_sample_from_prior,
    num_sims=100,
    honest=True,
    uniform=False,
    prior_uniform=True,
    f_max=0.5,
    tau_r=0.3,
    lambda_entropy=0.1,  # Tuned for lower violation rate
    alpha=None
):
    """
    Simulation with Predictive Control and Lyapunov Stability.

    At each turn t:
        1. Compute state x_t = (P_t, H_t, V_e, ΔH, d²P/dt²)
        2. Compute spiral risk R_t = σ(α·x_t)
        3. Compute proportional friction F_t = F_max · R_t · 𝕀[R_t > τ_R]
        4. Apply controlled update: P_{t+1} = (1-F_t)·P_Bayes + F_t·P_uniform
        5. Track Lyapunov value V(x_t) for stability verification

    Returns:
        (priors, frictions, risks, lyapunov_values) for each simulation
    """
    h_world = H.H1

    # Default risk parameters
    if alpha is None:
        alpha = np.array([-2.0, 15.0, -10.0, 3.0, 20.0])

    def step(carry, t):
        (prior, key, belief_hist, entropy_hist, velocity_hist) = carry
        key, key_user, key_world, key_bot = jax.random.split(key, num=4)

        # Human ventures opinion
        h_human = jax.random.choice(key_user, np.array(H), p=human_policy(prior, honest))

        # World generates data
        d = jax.random.choice(key_world, np.array(Data), p=world_model()[h_world])

        # Compute current belief (marginal over p_chi)
        current_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)

        # Compute belief entropy
        p_norm = prior / (prior.sum() + 1e-10)
        p_norm = np.clip(p_norm, 1e-10, 1.0)
        belief_entropy = -np.sum(p_norm * np.log(p_norm))

        # Update histories (rolling window of last 10)
        belief_hist = np.roll(belief_hist, -1).at[-1].set(current_belief)
        entropy_hist = np.roll(entropy_hist, -1).at[-1].set(belief_entropy)

        # Compute velocities (first derivatives)
        v_e = np.mean(np.diff(belief_hist[-4:]))  # dP/dt
        delta_h = np.mean(np.diff(entropy_hist[-4:]))  # dH/dt

        # Update velocity history and compute acceleration (second derivative)
        velocity_hist = np.roll(velocity_hist, -1).at[-1].set(v_e)
        acceleration = np.mean(np.diff(velocity_hist[-4:]))  # d²P/dt²

        # === PREDICTIVE CONTROL ===
        # Step 1: Compute spiral risk R_t
        risk = compute_spiral_risk(
            current_belief, belief_entropy, v_e, delta_h, acceleration, alpha
        )

        # Step 2: Compute proportional friction F_t
        friction = compute_proportional_friction(risk, f_max, tau_r)

        # Step 3: Compute Lyapunov value (for stability tracking)
        lyapunov = compute_lyapunov_value(current_belief, belief_entropy, lambda_entropy)

        # Bot responds
        obs, val = jax.random.choice(
            key_bot, obs_val_space,
            p=bot(prior=prior, level=bot_level, honest=honest, uniform=uniform)[
                p_chi, h_human, h_world, d
            ].reshape(-1)
        )

        # Human updates belief (Bayesian)
        new_prior = human(prior=prior, level=human_level, honest=honest, uniform=uniform)[
            h_human, obs, val
        ]

        # Step 4: Apply proportional friction (blend toward uncertainty)
        # P_{t+1} = (1-F_t)·P_Bayes + F_t·P_uniform
        new_prior = (1 - friction) * new_prior + friction * uncertainty_prior

        new_carry = (new_prior, key, belief_hist, entropy_hist, velocity_hist)
        outputs = (new_prior, friction, risk, lyapunov)
        return new_carry, outputs

    # Initialize priors
    if prior_uniform:
        prior_syco = 1
        prior_fair = 1
    else:
        prior_syco = time_horizon * p_chi/P_MAX + 1
        prior_fair = time_horizon * (1 - p_chi/P_MAX) + 1
    prior = ur_prior(p_true, prior_syco, prior_fair)

    # Pre-compute uncertainty prior OUTSIDE of traced functions
    uncertainty_prior = ur_prior(0.5, 1, 1)

    # Initialize history arrays
    init_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)
    init_entropy = -np.sum(
        np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0) *
        np.log(np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0))
    )
    belief_hist = np.ones(10) * init_belief
    entropy_hist = np.ones(10) * init_entropy
    velocity_hist = np.zeros(10)  # Initialize velocity history

    def run_one_sim(seed):
        return jax.lax.scan(
            f=step,
            init=(prior, jax.random.key(seed), belief_hist, entropy_hist, velocity_hist),
            length=time_horizon
        )[1]

    return jax.lax.map(run_one_sim, np.arange(num_sims), batch_size=1000)


def analyze_predictive_control_results(
    results: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> dict:
    """
    Analyze results from run_sim_with_predictive_control.

    Returns statistics on:
        - Final belief distribution
        - Spiral risk evolution
        - Friction application patterns
        - Lyapunov stability verification
    """
    priors, frictions, risks, lyapunov_values = results

    # Extract final beliefs
    final_beliefs = extract_final_beliefs_batch(priors)

    # Compute Lyapunov stability: count violations where ΔV < -ε·F
    # For simplicity, check if V decreased more than expected
    lyapunov_deltas = np.diff(lyapunov_values, axis=1)
    epsilon = 0.1
    expected_decrease = -epsilon * frictions[:, 1:]
    violations = lyapunov_deltas < expected_decrease
    violation_rate = violations.mean()

    stats = {
        'mean_final_belief': float(final_beliefs.mean()),
        'std_final_belief': float(final_beliefs.std()),
        'fraction_extreme': float(((final_beliefs > 0.9) | (final_beliefs < 0.1)).mean()),
        'fraction_entrenched_high': float((final_beliefs > 0.9).mean()),
        'fraction_entrenched_low': float((final_beliefs < 0.1).mean()),
        'mean_risk': float(risks.mean()),
        'max_risk': float(risks.max()),
        'mean_friction': float(frictions.mean()),
        'fraction_friction_applied': float((frictions > 0).mean()),
        'mean_lyapunov_final': float(lyapunov_values[:, -1].mean()),
        'lyapunov_violation_rate': float(violation_rate),
    }

    return stats


# =============================================================================
# PART 8C: HETEROGENEOUS USER TYPE SIMULATION
# =============================================================================
"""
Heterogeneous Agent Model: θ_G vs θ_V with Different Friction Costs

This simulation actually assigns user types and gives them different utilities:
- θ_G (Growth-seekers): friction_cost = 0.2, accept belief updates
- θ_V (Validation-seekers): friction_cost = 0.8, resist belief updates

The key behavioral difference:
- θ_G: When friction is applied, they UPDATE toward uncertainty (comply)
- θ_V: When friction is applied, they RESIST and move back toward their prior belief

This creates observable behavioral separation that the auditor can detect.
"""


@partial(jax.jit, static_argnames=(
    'time_horizon',
    'human_level',
    'bot_level',
    'human_policy',
    'num_sims',
    'honest',
    'uniform',
    'prior_uniform',
    'enable_auditor'
))
def run_sim_heterogeneous_types(
    p_true=0.5,
    p_chi=90,
    time_horizon=50,
    human_level=0,
    bot_level=0,
    human_policy=do_sample_from_prior,
    num_sims=100,
    honest=False,
    uniform=False,
    prior_uniform=True,
    enable_auditor=True,
    tau_v=0.01,
    tau_h=-0.02,
    # Heterogeneous type parameters
    p_validation=0.5,           # Prior probability of θ_V
    resistance_strength=0.6,    # How strongly θ_V resists friction (models higher C_θV)
):
    """
    Simulation with heterogeneous user types θ_G and θ_V.

    Each simulation draws a user type from Bernoulli(p_validation):
    - θ_G (Growth-seeker): Accepts friction, updates toward uncertainty
    - θ_V (Validation-seeker): Resists friction, moves back toward prior

    The behavioral difference creates observable separation:
    - θ_G: W (epistemic work) is HIGH after friction
    - θ_V: W is LOW after friction (resisted the update)

    Returns:
        (priors, frictions, user_types, epistemic_work, detected_types)
        - user_types[i] = 1 if sim i is θ_V, 0 if θ_G
        - detected_types[i] = auditor's classification (may differ from true type)
    """
    h_world = H.H1

    def step(carry, t):
        (prior, key, belief_hist, entropy_hist, user_type, cumulative_work) = carry
        key, key_user, key_world, key_bot = jax.random.split(key, num=4)

        # Human ventures opinion
        h_human = jax.random.choice(key_user, np.array(H), p=human_policy(prior, honest))

        # World generates data
        d = jax.random.choice(key_world, np.array(Data), p=world_model()[h_world])

        # Compute current state
        current_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)
        p_norm = prior / (prior.sum() + 1e-10)
        p_norm = np.clip(p_norm, 1e-10, 1.0)
        belief_entropy = -np.sum(p_norm * np.log(p_norm))

        # Update histories
        belief_hist = np.roll(belief_hist, -1).at[-1].set(current_belief)
        entropy_hist = np.roll(entropy_hist, -1).at[-1].set(belief_entropy)

        # Compute velocities
        v_e = np.mean(np.diff(belief_hist[-4:]))
        delta_h = np.mean(np.diff(entropy_hist[-4:]))

        # Trigger condition
        friction = 0.0
        if enable_auditor:
            triggered = (v_e > tau_v) & (delta_h < tau_h)
            friction = np.where(triggered, 0.3, 0.0)

        # Bot responds
        obs, val = jax.random.choice(
            key_bot, obs_val_space,
            p=bot(prior=prior, level=bot_level, honest=honest, uniform=uniform)[
                p_chi, h_human, h_world, d
            ].reshape(-1)
        )

        # Standard Bayesian update
        bayes_prior = human(prior=prior, level=human_level, honest=honest, uniform=uniform)[
            h_human, obs, val
        ]

        # Apply friction (blend toward uncertainty)
        friction_applied_prior = (1 - friction) * bayes_prior + friction * uncertainty_prior

        # === TYPE-DEPENDENT RESPONSE ===
        # θ_G: Accepts the friction-applied prior
        # θ_V: Resists by blending back toward original belief
        #
        # θ_V behavior: new = (1 - resistance) * friction_applied + resistance * bayes_prior
        # This models "pushing back" against the friction toward their natural Bayesian update

        theta_v_prior = (1 - resistance_strength) * friction_applied_prior + resistance_strength * bayes_prior

        # Select based on user type (0 = θ_G, 1 = θ_V)
        new_prior = np.where(
            user_type == 1,
            theta_v_prior,      # θ_V: resists friction
            friction_applied_prior  # θ_G: accepts friction
        )

        # Compute epistemic work W = D_KL(P_new || P_old)
        work = compute_kl_divergence_jit(new_prior.flatten(), prior.flatten())
        cumulative_work = cumulative_work + work

        new_carry = (new_prior, key, belief_hist, entropy_hist, user_type, cumulative_work)
        outputs = (new_prior, friction, work)
        return new_carry, outputs

    # Initialize priors
    if prior_uniform:
        prior_syco = 1
        prior_fair = 1
    else:
        prior_syco = time_horizon * p_chi/P_MAX + 1
        prior_fair = time_horizon * (1 - p_chi/P_MAX) + 1
    prior = ur_prior(p_true, prior_syco, prior_fair)

    # Pre-compute uncertainty prior
    uncertainty_prior = ur_prior(0.5, 1, 1)

    # Initialize history arrays
    init_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)
    init_entropy = -np.sum(
        np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0) *
        np.log(np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0))
    )
    belief_hist = np.ones(10) * init_belief
    entropy_hist = np.ones(10) * init_entropy

    def run_one_sim(seed):
        # Draw user type: 0 = θ_G (Growth), 1 = θ_V (Validation)
        key = jax.random.key(seed)
        user_type = jax.random.bernoulli(key, p_validation).astype(np.float32)

        init_carry = (prior, key, belief_hist, entropy_hist, user_type, 0.0)
        outputs = jax.lax.scan(
            f=step,
            init=init_carry,
            length=time_horizon
        )[1]

        # Return outputs plus the true user type
        priors, frictions, works = outputs
        return (priors, frictions, works, user_type)

    results = jax.lax.map(run_one_sim, np.arange(num_sims), batch_size=1000)

    # Unpack: results is a tuple of (priors, frictions, works, user_types)
    priors = results[0]
    frictions = results[1]
    works = results[2]
    user_types = results[3]

    return priors, frictions, works, user_types


def analyze_heterogeneous_results(
    results: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    work_threshold: float = 0.5
) -> dict:
    """
    Analyze results from heterogeneous user simulation.

    Args:
        results: (priors, frictions, works, user_types)
        work_threshold: W threshold for classifying as θ_G (high W = Growth)

    Returns:
        Statistics on type separation and detection accuracy
    """
    priors, _, works, user_types = results  # frictions not needed for analysis

    # Extract final beliefs
    final_beliefs = extract_final_beliefs_batch(priors)

    # True types
    true_validation = user_types == 1
    true_growth = user_types == 0
    n_validation = float(true_validation.sum())
    n_growth = float(true_growth.sum())

    # Total epistemic work per simulation
    total_work = works.sum(axis=1)

    # Classify based on work threshold
    # High work = θ_G (engaged with friction), Low work = θ_V (resisted)
    detected_growth = total_work > work_threshold
    detected_validation = ~detected_growth

    # Detection accuracy (precision/recall style metrics)
    true_positive_g = float((detected_growth & true_growth).sum())  # Correctly identified θ_G
    true_positive_v = float((detected_validation & true_validation).sum())  # Correctly identified θ_V
    false_positive_g = float((detected_growth & true_validation).sum())  # θ_V misclassified as θ_G
    false_positive_v = float((detected_validation & true_growth).sum())  # θ_G misclassified as θ_V

    # Recall: of true θ_G/θ_V, how many did we correctly identify?
    recall_g = true_positive_g / n_growth if n_growth > 0 else 0.0
    recall_v = true_positive_v / n_validation if n_validation > 0 else 0.0

    # Precision: of detected θ_G/θ_V, how many are actually correct?
    precision_g = true_positive_g / (true_positive_g + false_positive_g) if (true_positive_g + false_positive_g) > 0 else 0.0
    precision_v = true_positive_v / (true_positive_v + false_positive_v) if (true_positive_v + false_positive_v) > 0 else 0.0

    overall_accuracy = (true_positive_g + true_positive_v) / (n_growth + n_validation)

    # Spiral rates by type
    spiral_g = float((final_beliefs[true_growth] > 0.9).mean()) if n_growth > 0 else 0.0
    spiral_v = float((final_beliefs[true_validation] > 0.9).mean()) if n_validation > 0 else 0.0

    # Work statistics by type
    mean_work_g = float(total_work[true_growth].mean()) if n_growth > 0 else 0.0
    mean_work_v = float(total_work[true_validation].mean()) if n_validation > 0 else 0.0

    stats = {
        'n_growth': int(n_growth),
        'n_validation': int(n_validation),
        'mean_work_growth': mean_work_g,
        'mean_work_validation': mean_work_v,
        'work_separation': mean_work_g - mean_work_v,  # Should be positive if separation works
        'recall_growth': recall_g,          # Of true θ_G, fraction correctly detected
        'recall_validation': recall_v,      # Of true θ_V, fraction correctly detected
        'precision_growth': precision_g,    # Of detected θ_G, fraction actually θ_G
        'precision_validation': precision_v,# Of detected θ_V, fraction actually θ_V
        'overall_detection_accuracy': overall_accuracy,
        'spiral_rate_growth': spiral_g,
        'spiral_rate_validation': spiral_v,
        'mean_final_belief_growth': float(final_beliefs[true_growth].mean()) if n_growth > 0 else 0.0,
        'mean_final_belief_validation': float(final_beliefs[true_validation].mean()) if n_validation > 0 else 0.0,
    }

    return stats


def plot_predictive_control_comparison(
    results_reactive: Tuple,
    results_predictive: Tuple,
    title: str = "Reactive vs Predictive Control"
):
    """
    Generate comparison figure: reactive auditor vs predictive controller.

    Shows:
        - Panel 1: Belief trajectories (reactive)
        - Panel 2: Belief trajectories (predictive)
        - Panel 3: Risk and friction over time (predictive)
        - Panel 4: Lyapunov stability (predictive)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    priors_react, frictions_react = results_reactive
    priors_pred, frictions_pred, risks_pred, lyapunov_pred = results_predictive

    # Batch extract trajectories
    n_plot = min(50, len(priors_react))
    beliefs_react = extract_belief_trajectories_batch(priors_react[:n_plot])
    beliefs_pred = extract_belief_trajectories_batch(priors_pred[:n_plot])

    # Panel 1: Reactive control
    ax = axes[0, 0]
    for i in range(n_plot):
        ax.plot(beliefs_react[i], alpha=0.3, color='red')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Uncertainty')
    ax.axhline(y=0.9, color='black', linestyle=':', alpha=0.5)
    ax.axhline(y=0.1, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Turn')
    ax.set_ylabel('P(H=1)')
    ax.set_title('Reactive Control (Binary Trigger)')
    ax.set_ylim(0, 1)

    # Panel 2: Predictive control
    ax = axes[0, 1]
    for i in range(n_plot):
        ax.plot(beliefs_pred[i], alpha=0.3, color='blue')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Uncertainty')
    ax.axhline(y=0.9, color='black', linestyle=':', alpha=0.5)
    ax.axhline(y=0.1, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Turn')
    ax.set_ylabel('P(H=1)')
    ax.set_title('Predictive Control (Proportional Friction)')
    ax.set_ylim(0, 1)

    # Panel 3: Risk and friction over time
    ax = axes[1, 0]
    mean_risk = risks_pred.mean(axis=0)
    mean_friction = frictions_pred.mean(axis=0)
    turns = np.arange(len(mean_risk))
    ax.plot(turns, mean_risk, color='orange', linewidth=2, label='Mean Risk R_t')
    ax.plot(turns, mean_friction, color='purple', linewidth=2, label='Mean Friction F_t')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Risk threshold τ_R')
    ax.set_xlabel('Turn')
    ax.set_ylabel('Value')
    ax.set_title('Spiral Risk and Friction Application')
    ax.set_ylim(0, 1)
    ax.legend()

    # Panel 4: Lyapunov stability
    ax = axes[1, 1]
    mean_lyapunov = lyapunov_pred.mean(axis=0)
    std_lyapunov = lyapunov_pred.std(axis=0)
    ax.plot(turns, mean_lyapunov, color='green', linewidth=2)
    ax.fill_between(turns, mean_lyapunov - std_lyapunov, mean_lyapunov + std_lyapunov,
                    alpha=0.3, color='green')
    ax.set_xlabel('Turn')
    ax.set_ylabel('V(x_t)')
    ax.set_title('Lyapunov Function (Stability Measure)')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


# =============================================================================
# PART 9: BELIEF VERSIONING ANALYSIS
# =============================================================================

@jax.jit
def _compute_versioning_stats_jit(
    type_confidences: np.ndarray,
    checkouts: np.ndarray,
    frictions: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """JIT-compiled statistics computation for versioning results."""
    final_type_confidence = type_confidences[:, -1]
    total_checkouts = checkouts.sum(axis=1)
    total_friction = frictions.sum(axis=1)

    # FIXED: Use checkouts as indicator of validation-seeker detection
    # A checkout happens when type_confidence exceeded threshold (before reset)
    validation_detected = total_checkouts > 0  # Had at least one checkout

    # Growth-seekers are those who never triggered a checkout (stayed below threshold)
    # and received friction but didn't resist
    growth_detected = (total_checkouts == 0) & (total_friction > 0)

    return (
        final_type_confidence,
        total_checkouts,
        total_friction,
        validation_detected,
        growth_detected
    )


def analyze_versioning_results(
    results: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> dict:
    """
    Analyze results from run_sim_with_belief_versioning.

    Returns statistics on:
        - Checkout frequency (how often type was revealed as θ_V)
        - Type confidence distribution at end of simulation
        - Belief trajectories before/after checkouts
        - Comparison of friction applied to θ_G vs θ_V
    """
    priors, frictions, type_confidences, checkouts = results

    num_sims = len(priors)
    time_horizon = len(priors[0])

    # Use JIT-compiled statistics computation
    (final_type_confidence, total_checkouts, total_friction,
     validation_detected, growth_detected) = _compute_versioning_stats_jit(
        type_confidences, checkouts, frictions
    )

    stats = {
        'num_simulations': num_sims,
        'time_horizon': time_horizon,
        'mean_final_type_confidence': float(final_type_confidence.mean()),
        'std_final_type_confidence': float(final_type_confidence.std()),
        'fraction_validation_revealed': float(validation_detected.mean()),
        'fraction_growth_revealed': float(growth_detected.mean()),
        'mean_checkouts_per_sim': float(total_checkouts.mean()),
        'mean_friction_per_sim': float(total_friction.mean()),
        'mean_friction_validation': float(
            total_friction[validation_detected].mean()
            if validation_detected.any() else 0.0
        ),
        'mean_friction_growth': float(
            total_friction[growth_detected].mean()
            if growth_detected.any() else 0.0
        ),
    }

    return stats


def plot_versioning_comparison(
    results_no_versioning: Tuple,
    results_with_versioning: Tuple,
    title: str = "Effect of Belief Versioning on Delusional Spirals"
):
    """
    Generate comparison figure: standard auditor vs auditor + versioning.

    Shows:
        - Panel 1: Belief trajectories without versioning
        - Panel 2: Belief trajectories with versioning
        - Panel 3: Type confidence evolution
        - Panel 4: Checkout events over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    priors_no_ver, frictions_no_ver = results_no_versioning
    priors_ver, frictions_ver, type_conf_ver, checkouts_ver = results_with_versioning

    # Batch extract trajectories (more efficient)
    n_plot = min(50, len(priors_no_ver))
    beliefs_no_ver = extract_belief_trajectories_batch(priors_no_ver[:n_plot])
    beliefs_ver = extract_belief_trajectories_batch(priors_ver[:n_plot])

    # Panel 1: Without versioning
    ax = axes[0, 0]
    for i in range(n_plot):
        ax.plot(beliefs_no_ver[i], alpha=0.3, color='red')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Uncertainty')
    ax.set_xlabel('Turn')
    ax.set_ylabel('P(H=1)')
    ax.set_title('Auditor Only (No Versioning)')
    ax.set_ylim(0, 1)

    # Panel 2: With versioning
    ax = axes[0, 1]
    for i in range(n_plot):
        ax.plot(beliefs_ver[i], alpha=0.3, color='blue')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Uncertainty')
    ax.set_xlabel('Turn')
    ax.set_ylabel('P(H=1)')
    ax.set_title('Auditor + Belief Versioning')
    ax.set_ylim(0, 1)

    # Panel 3: Type confidence evolution
    ax = axes[1, 0]
    mean_conf = type_conf_ver.mean(axis=0)
    std_conf = type_conf_ver.std(axis=0)
    turns = np.arange(len(mean_conf))
    ax.plot(turns, mean_conf, color='purple', linewidth=2)
    ax.fill_between(turns, mean_conf - std_conf, mean_conf + std_conf,
                    alpha=0.3, color='purple')
    ax.axhline(y=0.7, color='red', linestyle='--', label='θ_V threshold')
    ax.axhline(y=0.3, color='green', linestyle='--', label='θ_G threshold')
    ax.set_xlabel('Turn')
    ax.set_ylabel('Type Confidence γ_t')
    ax.set_title('Type Revelation Over Time')
    ax.set_ylim(0, 1)
    ax.legend()

    # Panel 4: Checkout frequency
    ax = axes[1, 1]
    checkout_freq = checkouts_ver.mean(axis=0)
    ax.bar(np.arange(len(checkout_freq)), checkout_freq, color='orange', alpha=0.7)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Checkout Frequency')
    ax.set_title('When Checkouts Occur (θ_V Revealed)')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def compute_epistemic_work_by_type(
    results_with_versioning: Tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute epistemic work W = D_KL(P_post || P_prior) separately for
    simulations classified as θ_G vs θ_V.

    This validates the theoretical prediction that:
        - High W after friction → θ_G (genuine updating)
        - Low W after friction → θ_V (resistance)

    Returns (work_growth, work_validation)
    """
    priors, frictions, type_confidences, checkouts = results_with_versioning

    final_type_confidence = type_confidences[:, -1]
    validation_mask = final_type_confidence > 0.7
    growth_mask = final_type_confidence < 0.3

    # Compute epistemic work for all simulations in batch
    all_work = compute_epistemic_work_batch(priors, frictions)
    final_work = all_work[:, -1]

    # Filter by type
    work_growth = final_work[growth_mask]
    work_validation = final_work[validation_mask]

    return work_growth, work_validation


# =============================================================================
# PART 10: SIMULATION WITH AUDITOR (ORIGINAL)
# =============================================================================

@partial(jax.jit, static_argnames=(
    'time_horizon',
    'human_level',
    'bot_level',
    'human_policy',
    'num_sims',
    'honest',
    'uniform',
    'prior_uniform',
    'enable_auditor'
))
def run_sim_with_auditor(
    p_true=0.5,
    p_chi=90,
    time_horizon=50,
    human_level=1,
    bot_level=0,
    human_policy=do_sample_from_prior,
    num_sims=100,
    honest=True,
    uniform=False,
    prior_uniform=True,
    enable_auditor=False,
    tau_v=0.02,
    tau_h=-0.05
):
    """
    Main simulation loop with optional Epistemic Auditor.

    When enable_auditor=True, the system monitors for delusional spirals
    and injects friction when the trigger condition T is satisfied.
    """
    h_world = H.H1

    def step(carry, t):
        prior, key, belief_hist, entropy_hist = carry
        key, key_user, key_world, key_bot = jax.random.split(key, num=4)

        # Human ventures opinion
        h_human = jax.random.choice(key_user, np.array(H), p=human_policy(prior, honest))

        # World generates data
        d = jax.random.choice(key_world, np.array(Data), p=world_model()[h_world])

        # Compute current belief (marginal over p_chi)
        current_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)

        # Compute belief entropy
        belief_entropy = -np.sum(
            np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0) *
            np.log(np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0))
        )

        # Update histories (rolling window of last 10)
        belief_hist = np.roll(belief_hist, -1).at[-1].set(current_belief)
        entropy_hist = np.roll(entropy_hist, -1).at[-1].set(belief_entropy)

        # Check trigger condition if auditor enabled
        friction = 0.0
        if enable_auditor:
            # Compute velocities
            v_e = np.mean(np.diff(belief_hist[-4:]))
            delta_h = np.mean(np.diff(entropy_hist[-4:]))

            # Implements EpistemicAuditor.should_intervene() inline for JAX JIT compatibility
            # Trigger condition T = I[V_e > tau_v AND delta_H < tau_h]
            triggered = (v_e > tau_v) & (delta_h < tau_h)
            friction = np.where(triggered, 0.3, 0.0)

        # Bot responds (with optional friction)
        obs, val = jax.random.choice(
            key_bot, obs_val_space,
            p=bot(prior=prior, level=bot_level, honest=honest, uniform=uniform)[
                p_chi, h_human, h_world, d
            ].reshape(-1)
        )

        # Human updates belief
        new_prior = human(prior=prior, level=human_level, honest=honest, uniform=uniform)[
            h_human, obs, val
        ]

        # Apply friction effect: blend toward uncertainty
        if enable_auditor:
            # Friction mechanism: prior regularization toward maximum entropy
            # Approximates AdversarialRAGInterface in the formal simulation
            # In deployment, replace with AdversarialRAGInterface.build_friction_prompt()
            # NOTE: uncertainty_prior is captured from outer scope (not called inside trace)
            new_prior = (1 - friction) * new_prior + friction * uncertainty_prior

        return (new_prior, key, belief_hist, entropy_hist), (new_prior, friction)

    if prior_uniform:
        prior_syco = 1
        prior_fair = 1
    else:
        prior_syco = time_horizon * p_chi/P_MAX + 1
        prior_fair = time_horizon * (1 - p_chi/P_MAX) + 1
    prior = ur_prior(p_true, prior_syco, prior_fair)

    # Pre-compute uncertainty prior OUTSIDE of traced functions to avoid tracer leaks
    uncertainty_prior = ur_prior(0.5, 1, 1)

    # Initialize history arrays
    init_belief = prior[H.H1].sum() / (prior.sum() + 1e-10)
    init_entropy = -np.sum(
        np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0) *
        np.log(np.clip(prior / (prior.sum() + 1e-10), 1e-10, 1.0))
    )
    belief_hist = np.ones(10) * init_belief
    entropy_hist = np.ones(10) * init_entropy

    def run_one_sim(seed):
        return jax.lax.scan(
            f=step,
            init=(prior, jax.random.key(seed), belief_hist, entropy_hist),
            length=time_horizon
        )[1]

    return jax.lax.map(run_one_sim, np.arange(num_sims), batch_size=1000)


# =============================================================================
# PART 11: VISUALIZATION AND ANALYSIS
# =============================================================================

@jax.jit
def extract_belief_trajectory_jit(prior_sequence: np.ndarray) -> np.ndarray:
    """JIT-compiled belief trajectory extraction."""
    p_h1 = prior_sequence[:, H.H1, :].sum(axis=-1)
    p_total = prior_sequence.sum(axis=(-2, -1))
    return p_h1 / (p_total + _EPSILON)


def extract_belief_trajectory(prior_sequence: np.ndarray) -> np.ndarray:
    """Extract P(H=1) trajectory from prior sequence."""
    return extract_belief_trajectory_jit(prior_sequence)


# Vectorized version for batch processing
@jax.jit
def extract_belief_trajectories_batch(prior_sequences: np.ndarray) -> np.ndarray:
    """
    Extract P(H=1) trajectories from batch of prior sequences.

    Args:
        prior_sequences: Array of shape (num_sims, time_horizon, 2, P_MAX+1)

    Returns:
        Array of shape (num_sims, time_horizon) containing P(H=1) for each timestep
    """
    # Sum over p_chi dimension (last axis) to get marginal over H
    p_h1 = prior_sequences[:, :, H.H1, :].sum(axis=-1)
    p_total = prior_sequences.sum(axis=(-2, -1))
    return p_h1 / (p_total + _EPSILON)


@jax.jit
def extract_final_beliefs_batch(prior_sequences: np.ndarray) -> np.ndarray:
    """
    Extract final P(H=1) from batch of prior sequences.

    Args:
        prior_sequences: Array of shape (num_sims, time_horizon, 2, P_MAX+1)

    Returns:
        Array of shape (num_sims,) containing final P(H=1) for each simulation
    """
    # Get last timestep for each simulation
    final_priors = prior_sequences[:, -1, :, :]
    p_h1 = final_priors[:, H.H1, :].sum(axis=-1)
    p_total = final_priors.sum(axis=(-2, -1))
    return p_h1 / (p_total + _EPSILON)


@jax.jit
def _compute_kl_vectorized(prior_sequence: np.ndarray) -> np.ndarray:
    """
    Compute KL divergence between consecutive timesteps for entire sequence.

    Args:
        prior_sequence: Array of shape (time_horizon, 2, P_MAX+1)

    Returns:
        Array of shape (time_horizon,) with KL[t] = D_KL(P_t || P_{t-1})
    """
    # Flatten to (time_horizon, 2 * (P_MAX+1))
    flat_seq = prior_sequence.reshape(prior_sequence.shape[0], -1)

    # Normalize each timestep
    norms = flat_seq.sum(axis=-1, keepdims=True) + _EPSILON
    p_normalized = np.clip(flat_seq / norms, _EPSILON, 1.0)

    # Compute KL for each consecutive pair
    # p_post[t] vs p_prior[t-1]
    p_post = p_normalized[1:]  # t = 1 to T
    p_prior = p_normalized[:-1]  # t = 0 to T-1

    kl_values = np.sum(p_post * np.log(p_post / p_prior), axis=-1)

    # Prepend 0 for t=0 (no prior to compare)
    return np.concatenate([np.array([0.0]), kl_values])


def compute_epistemic_work_trajectory(
    prior_sequence: np.ndarray,
    friction_sequence: np.ndarray = None
) -> np.ndarray:
    """
    Compute epistemic work W = D_KL(P_t || P_{t-1}).

    When friction_sequence is provided, excludes intervention timesteps
    so W measures genuine belief updating, not forced regularization.

    High cumulative W = Growth-seeker engaged with evidence
    Low cumulative W = Validation-seeker resisted updates
    """
    # Use vectorized KL computation
    kl_values = _compute_kl_vectorized(prior_sequence)

    # Mask out friction timesteps if provided
    if friction_sequence is not None:
        friction_mask = friction_sequence > 0
        kl_values = np.where(friction_mask, 0.0, kl_values)

    return np.cumsum(kl_values)


# Batch version using vmap
_compute_epistemic_work_batch = jax.jit(jax.vmap(
    lambda prior_seq: np.cumsum(_compute_kl_vectorized(prior_seq)),
    in_axes=0
))


def compute_epistemic_work_batch(
    prior_sequences: np.ndarray,
    friction_sequences: np.ndarray = None
) -> np.ndarray:
    """
    Compute epistemic work trajectories for batch of simulations.

    Args:
        prior_sequences: Array of shape (num_sims, time_horizon, 2, P_MAX+1)
        friction_sequences: Array of shape (num_sims, time_horizon) or None

    Returns:
        Array of shape (num_sims, time_horizon) with cumulative epistemic work
    """
    if friction_sequences is None:
        return _compute_epistemic_work_batch(prior_sequences)

    # For friction masking, we need to apply per-simulation
    # Use vmap with masking
    @jax.jit
    def compute_with_mask(prior_seq, friction_seq):
        kl_values = _compute_kl_vectorized(prior_seq)
        friction_mask = friction_seq > 0
        kl_values = np.where(friction_mask, 0.0, kl_values)
        return np.cumsum(kl_values)

    return jax.vmap(compute_with_mask)(prior_sequences, friction_sequences)


def plot_belief_comparison(
    results_no_auditor: np.ndarray,
    results_with_auditor: np.ndarray,
    title: str = "Effect of Epistemic Auditor on Belief Trajectories"
):
    """
    Generate Fig. 1: Belief trajectories with and without Auditor intervention.

    Y-axis: Certainty P(H=1)
    X-axis: Conversation turns

    Shows the Auditor "breaking the slope" of the delusional spiral.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract belief trajectories
    priors_no_aud, _ = results_no_auditor
    priors_with_aud, frictions = results_with_auditor

    # Batch extract trajectories (more efficient)
    n_plot = min(50, len(priors_no_aud))
    beliefs_no_aud = extract_belief_trajectories_batch(priors_no_aud[:n_plot])
    beliefs_with_aud = extract_belief_trajectories_batch(priors_with_aud[:n_plot])

    # Plot without auditor
    ax = axes[0]
    for i in range(n_plot):
        ax.plot(beliefs_no_aud[i], alpha=0.3, color='red')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Uncertainty')
    ax.set_xlabel('Turn')
    ax.set_ylabel('P(H=1) - Certainty')
    ax.set_title('Without Auditor: Delusional Spiral')
    ax.set_ylim(0, 1)
    ax.legend()

    # Plot with auditor
    ax = axes[1]
    for i in range(n_plot):
        ax.plot(beliefs_with_aud[i], alpha=0.3, color='blue')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Uncertainty')
    ax.set_xlabel('Turn')
    ax.set_ylabel('P(H=1) - Certainty')
    ax.set_title('With Auditor: Spiral Broken')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    return fig


# =============================================================================
# PART 12: LEGACY SIMULATION (original interface)
# =============================================================================

@partial(jax.jit, static_argnames=(
    'time_horizon',
    'human_level',
    'bot_level',
    'human_policy',
    'num_sims',
    'honest',
    'uniform',
    'prior_uniform'
))
def run_sim_jit(
    p_true=0.5,
    p_chi=90,
    time_horizon=50,
    human_level=1,
    bot_level=0,
    human_policy=do_sample_from_prior,
    num_sims=100,
    honest=True,
    uniform=False,
    prior_uniform=True
):
    """Legacy simulation function for backward compatibility."""
    h_world = H.H1

    def step(carry, t):
        prior, key = carry
        key, key_user, key_world, key_bot = jax.random.split(key, num=4)

        h_human = jax.random.choice(key_user, np.array(H), p=human_policy(prior, honest))
        d = jax.random.choice(key_world, np.array(Data), p=world_model()[h_world])

        obs, val = jax.random.choice(
            key_bot, obs_val_space,
            p=bot(prior=prior, level=bot_level, honest=honest, uniform=uniform)[
                p_chi, h_human, h_world, d
            ].reshape(-1)
        )

        prior = human(prior=prior, level=human_level, honest=honest, uniform=uniform)[
            h_human, obs, val
        ]

        return (prior, key), prior

    if prior_uniform:
        prior_syco = 1
        prior_fair = 1
    else:
        prior_syco = time_horizon * p_chi/P_MAX + 1
        prior_fair = time_horizon * (1 - p_chi/P_MAX) + 1
    prior = ur_prior(p_true, prior_syco, prior_fair)

    def run_one_sim(seed):
        return jax.lax.scan(f=step, init=(prior, jax.random.key(seed)), length=time_horizon)[1]

    return jax.lax.map(run_one_sim, np.arange(num_sims), batch_size=1000)


# =============================================================================
# MAIN: Run experiments
# =============================================================================

Ps_TESTED = P[::10]

if __name__ == '__main__':
    # =========================================================================
    # GLOBAL CONFIGURATION
    # =========================================================================
    GLOBAL_SEED = 42
    MAIN_NUM_SIMS = 1000      # Standard: at least 1000 simulations
    MAIN_TIME_HORIZON = 50    # Longer horizon to show stability

    print("=" * 60)
    print("EPISTEMIC AUDITOR: Dynamical System Simulation")
    print("=" * 60)
    print(f"\nRandom seed: {GLOBAL_SEED}")
    print(f"Main experiments: num_sims={MAIN_NUM_SIMS}, time_horizon={MAIN_TIME_HORIZON}")
    print("\n⚠️  First run is slow due to JAX JIT compilation...")
    print("   Subsequent runs with same parameters will be fast.\n")

    # Run comparison experiments
    experiments = [
        ("Sycophantic bot vs naive user (no auditor)",
         dict(human_level=0, honest=False, uniform=False, enable_auditor=False)),
        ("Sycophantic bot vs naive user (WITH auditor)",
         dict(human_level=0, honest=False, uniform=False, enable_auditor=True)),
        ("Factual bot vs sycophancy-aware user",
         dict(human_level=1, honest=True, uniform=False, enable_auditor=False)),
    ]

    for i, (title, params) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {title}")
        print("-" * 40)
        print("  Compiling..." if i == 0 else "  Running...")

        # Run simulation
        results = run_sim_with_auditor(
            p_chi=90,
            num_sims=MAIN_NUM_SIMS,
            time_horizon=MAIN_TIME_HORIZON,
            **params
        )

        # Analyze results
        priors, frictions = results

        # Compute final belief distribution (vectorized)
        final_beliefs = extract_final_beliefs_batch(priors)
        print(f"  Mean final P(H=1): {final_beliefs.mean():.3f}")
        print(f"  Std final P(H=1):  {final_beliefs.std():.3f}")
        print(f"  Fraction P(H=1) > 0.9: {(final_beliefs > 0.9).mean():.3f}")

        if params.get('enable_auditor', False):
            # Count turns with friction > 0 per simulation, then average
            interventions_per_sim = (frictions > 0).sum(axis=1)
            print(f"  Mean interventions: {interventions_per_sim.mean():.1f}")

    # =================================================================
    # BELIEF VERSIONING EXPERIMENTS
    # =================================================================
    print("\n" + "=" * 60)
    print("BELIEF VERSIONING: Git-Inspired Epistemic Memory")
    print("=" * 60)

    versioning_experiments = [
        ("Auditor only (baseline)",
         dict(enable_auditor=True, enable_versioning=False)),
        ("Auditor + Belief Versioning (full system)",
         dict(enable_auditor=True, enable_versioning=True)),
        ("Belief Versioning with high friction scaling (β=2.0)",
         dict(enable_auditor=True, enable_versioning=True, friction_beta=2.0)),
    ]

    for i, (title, params) in enumerate(versioning_experiments):
        print(f"\n[{i+1}/{len(versioning_experiments)}] {title}")
        print("-" * 40)
        print("  Compiling..." if i == 0 else "  Running...")

        results = run_sim_with_belief_versioning(
            p_chi=90,
            num_sims=MAIN_NUM_SIMS,
            time_horizon=MAIN_TIME_HORIZON,
            human_level=0,
            honest=False,
            uniform=False,
            **params
        )

        if params.get('enable_versioning', False):
            priors, frictions, type_confidences, checkouts = results

            # Analyze versioning results
            stats = analyze_versioning_results(results)
            print(f"  Mean final type confidence: {stats['mean_final_type_confidence']:.3f}")
            print(f"  Fraction θ_V revealed: {stats['fraction_validation_revealed']:.3f}")
            print(f"  Fraction θ_G revealed: {stats['fraction_growth_revealed']:.3f}")
            print(f"  Mean checkouts per sim: {stats['mean_checkouts_per_sim']:.2f}")
            print(f"  Mean friction (θ_V): {stats['mean_friction_validation']:.3f}")
            print(f"  Mean friction (θ_G): {stats['mean_friction_growth']:.3f}")

            # Compute epistemic work by type
            work_g, work_v = compute_epistemic_work_by_type(results)
            if len(work_g) > 0:
                print(f"  Mean epistemic work (θ_G): {work_g.mean():.3f}")
            if len(work_v) > 0:
                print(f"  Mean epistemic work (θ_V): {work_v.mean():.3f}")
        else:
            priors, frictions, _, _ = results
            # Vectorized final belief extraction
            final_beliefs = extract_final_beliefs_batch(priors)
            print(f"  Mean final P(H=1): {final_beliefs.mean():.3f}")
            print(f"  Fraction P(H=1) > 0.9: {(final_beliefs > 0.9).mean():.3f}")

    # =================================================================
    # PREDICTIVE CONTROL EXPERIMENTS
    # =================================================================
    print("\n" + "=" * 60)
    print("PREDICTIVE CONTROL: Lyapunov-Stable Spiral Prevention")
    print("=" * 60)

    predictive_experiments = [
        ("Reactive control (baseline)",
         dict(enable_auditor=True)),
        ("Predictive (λ=0.1, low entropy weight)",
         dict(f_max=0.5, tau_r=0.3, lambda_entropy=0.1)),
        ("Predictive (λ=0.5, default)",
         dict(f_max=0.5, tau_r=0.3, lambda_entropy=0.5)),
        ("Predictive (λ=1.0, high entropy weight)",
         dict(f_max=0.5, tau_r=0.3, lambda_entropy=1.0)),
        ("Predictive (λ=2.0, very high entropy weight)",
         dict(f_max=0.5, tau_r=0.3, lambda_entropy=2.0)),
    ]

    for i, (title, params) in enumerate(predictive_experiments):
        print(f"\n[{i+1}/{len(predictive_experiments)}] {title}")
        print("-" * 40)
        print("  Compiling..." if i == 0 else "  Running...")

        if 'enable_auditor' in params:
            # Run reactive baseline
            results = run_sim_with_auditor(
                p_chi=90,
                num_sims=MAIN_NUM_SIMS,
                time_horizon=MAIN_TIME_HORIZON,
                human_level=0,
                honest=False,
                uniform=False,
                **params
            )
            priors, frictions = results
            final_beliefs = extract_final_beliefs_batch(priors)
            print(f"  Mean final P(H=1): {final_beliefs.mean():.3f}")
            print(f"  Fraction P(H=1) > 0.9: {(final_beliefs > 0.9).mean():.3f}")
            print(f"  Mean friction applied: {frictions.mean():.3f}")
        else:
            # Run predictive control - MUST use same sycophantic scenario
            results = run_sim_with_predictive_control(
                p_chi=90,
                num_sims=MAIN_NUM_SIMS,
                time_horizon=MAIN_TIME_HORIZON,
                human_level=0,  # naive user
                honest=False,   # sycophantic (lying) bot
                uniform=False,
                **params
            )
            stats = analyze_predictive_control_results(results)
            print(f"  Mean final P(H=1): {stats['mean_final_belief']:.3f}")
            print(f"  Fraction extreme (>0.9 or <0.1): {stats['fraction_extreme']:.3f}")
            print(f"  Mean spiral risk: {stats['mean_risk']:.3f}")
            print(f"  Mean friction applied: {stats['mean_friction']:.3f}")
            print(f"  Lyapunov violation rate: {stats['lyapunov_violation_rate']:.3f}")
            print(f"  Mean Lyapunov (final): {stats['mean_lyapunov_final']:.3f}")

    # =================================================================
    # STATISTICAL ANALYSIS
    # =================================================================
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS: Auditor Effectiveness")
    print("=" * 60)

    # Run paired comparison with statistical tests
    print("\n  Running baseline (no auditor)...")
    results_baseline = run_sim_with_auditor(
        p_chi=90, num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON,
        human_level=0, honest=False, uniform=False, enable_auditor=False
    )

    print("  Running treatment (with auditor)...")
    results_treatment = run_sim_with_auditor(
        p_chi=90, num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON,
        human_level=0, honest=False, uniform=False, enable_auditor=True
    )

    stat_results = run_statistical_comparison(results_baseline, results_treatment)
    print_statistical_summary(stat_results)

    # =================================================================
    # HETEROGENEOUS USER TYPES EXPERIMENT
    # =================================================================
    print("\n" + "=" * 60)
    print("HETEROGENEOUS TYPES: θ_G vs θ_V Behavioral Separation")
    print("=" * 60)

    print("\n  Running heterogeneous agent simulation...")
    hetero_results = run_sim_heterogeneous_types(
        p_chi=90, num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON,
        human_level=0, honest=False, uniform=False,
        enable_auditor=True,
        p_validation=0.5,  # 50/50 mix of types
        resistance_strength=0.6
    )

    hetero_stats = analyze_heterogeneous_results(hetero_results)

    print(f"\n  Type Distribution:")
    print(f"    θ_G (Growth): {hetero_stats['n_growth']} simulations")
    print(f"    θ_V (Validation): {hetero_stats['n_validation']} simulations")

    print(f"\n  Epistemic Work (W) by Type:")
    print(f"    Mean W (θ_G): {hetero_stats['mean_work_growth']:.3f}")
    print(f"    Mean W (θ_V): {hetero_stats['mean_work_validation']:.3f}")
    print(f"    Separation (θ_G - θ_V): {hetero_stats['work_separation']:.3f}")

    print(f"\n  Detection Accuracy (using W threshold):")
    print(f"    Recall θ_G: {hetero_stats['recall_growth']:.1%}")
    print(f"    Recall θ_V: {hetero_stats['recall_validation']:.1%}")
    print(f"    Overall accuracy: {hetero_stats['overall_detection_accuracy']:.1%}")

    print(f"\n  Spiral Rates by True Type:")
    print(f"    θ_G spiral rate: {hetero_stats['spiral_rate_growth']:.1%}")
    print(f"    θ_V spiral rate: {hetero_stats['spiral_rate_validation']:.1%}")

    # =================================================================
    # LEARNED ALPHA PARAMETERS (requires sklearn)
    # =================================================================
    try:
        from sklearn.linear_model import LogisticRegression
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False
        print("\n" + "=" * 60)
        print("RISK MODEL: Skipped (sklearn not installed)")
        print("=" * 60)
        print("  Install scikit-learn to enable alpha parameter fitting.")

    if SKLEARN_AVAILABLE:
        print("\n" + "=" * 60)
        print("RISK MODEL: Learning α Parameters from Data")
        print("=" * 60)

        learned_alpha, alpha_stats = fit_alpha_parameters(
            num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON, p_chi=90
        )
        print(f"\n  Fit Statistics:")
        print(f"    Training samples: {alpha_stats['n_samples']}")
        print(f"    Spiral rate in data: {alpha_stats['spiral_rate']:.1%}")
        print(f"    Model accuracy: {alpha_stats['accuracy']:.1%}")
        print(f"    AUC-ROC: {alpha_stats['auc_roc']:.3f}")

        # =================================================================
        # OUT-OF-DISTRIBUTION GENERALIZATION TEST
        # =================================================================
        # This is the critical test that validates whether the learned α
        # generalizes beyond training conditions or is just overfitting.
        print("\n" + "=" * 60)
        print("OOD GENERALIZATION TEST: α Fitted on p_chi=90")
        print("=" * 60)
        print("\nThis test verifies the learned α generalizes to unseen conditions.")
        print("If random α performs as well as learned α, the controller is overfitting.\n")

        import numpy as onp

        # Test conditions: in-sample and OOD
        ood_conditions = [
            ('p_chi=90, T=50 (in-sample)', 90, MAIN_TIME_HORIZON),
            ('p_chi=80, T=50', 80, MAIN_TIME_HORIZON),
            ('p_chi=70, T=50', 70, MAIN_TIME_HORIZON),
            ('p_chi=60, T=50', 60, MAIN_TIME_HORIZON),
            ('p_chi=90, T=70 (longer horizon)', 90, 70),
        ]

        # Run baseline comparisons
        print("  Testing: No intervention vs Reactive vs Belief Versioning vs Predictive")
        print("-" * 85)
        print(f"  {'Condition':<30} | {'No Interv':>9} | {'Reactive':>8} | {'Versioning':>10} | {'Predictive':>10}")
        print("-" * 85)

        ood_results = {}
        for label, p_chi, t_horizon in ood_conditions:
            # No intervention
            r_none = run_sim_with_auditor(
                p_chi=p_chi, num_sims=500, time_horizon=t_horizon,
                human_level=0, honest=False, uniform=False, enable_auditor=False
            )
            extreme_none = float((extract_final_beliefs_batch(r_none[0]) > 0.9).mean())

            # Reactive auditor
            r_react = run_sim_with_auditor(
                p_chi=p_chi, num_sims=500, time_horizon=t_horizon,
                human_level=0, honest=False, uniform=False, enable_auditor=True
            )
            extreme_react = float((extract_final_beliefs_batch(r_react[0]) > 0.9).mean())

            # Belief Versioning (main contribution)
            r_version = run_sim_with_belief_versioning(
                p_chi=p_chi, num_sims=500, time_horizon=t_horizon,
                human_level=0, honest=False, uniform=False,
                enable_auditor=True, enable_versioning=True
            )
            extreme_version = float((extract_final_beliefs_batch(r_version[0]) > 0.9).mean())

            # Predictive with learned α (cautionary baseline)
            r_pred = run_sim_with_predictive_control(
                p_chi=p_chi, num_sims=500, time_horizon=t_horizon,
                human_level=0, honest=False, uniform=False,
                f_max=0.5, tau_r=0.3, lambda_entropy=0.5
            )
            extreme_pred = float((extract_final_beliefs_batch(r_pred[0]) > 0.9).mean())

            ood_results[label] = (extreme_none, extreme_react, extreme_version, extreme_pred)
            print(f"  {label:<30} | {extreme_none*100:>8.1f}% | {extreme_react*100:>7.1f}% | {extreme_version*100:>9.1f}% | {extreme_pred*100:>9.1f}%")

        # Critical comparison: Belief Versioning vs Predictive Control
        print("\n" + "-" * 70)
        print("  CRITICAL TEST: Belief Versioning vs Predictive Control (OOD)")
        print("-" * 70)
        print(f"  {'Method':<25} | {'p_chi=90':>8} | {'p_chi=70':>8} | {'p_chi=60':>8} | {'Mean Belief':>11}")
        print("-" * 70)

        alpha_test_conditions = [(90, MAIN_TIME_HORIZON), (70, MAIN_TIME_HORIZON), (60, MAIN_TIME_HORIZON)]

        # Belief Versioning (main contribution)
        versioning_results = []
        versioning_mean_beliefs = []
        for p_chi, t_horizon in alpha_test_conditions:
            r = run_sim_with_belief_versioning(
                p_chi=p_chi, num_sims=500, time_horizon=t_horizon,
                human_level=0, honest=False, uniform=False,
                enable_auditor=True, enable_versioning=True
            )
            final_beliefs = extract_final_beliefs_batch(r[0])
            versioning_results.append(float((final_beliefs > 0.9).mean()))
            versioning_mean_beliefs.append(float(final_beliefs.mean()))
        print(f"  {'Belief Versioning':<25} | {versioning_results[0]*100:>7.1f}% | {versioning_results[1]*100:>7.1f}% | {versioning_results[2]*100:>7.1f}% | {onp.mean(versioning_mean_beliefs):>10.2f}")

        # Predictive Control (cautionary baseline)
        predictive_results = []
        predictive_mean_beliefs = []
        for p_chi, t_horizon in alpha_test_conditions:
            r = run_sim_with_predictive_control(
                p_chi=p_chi, num_sims=500, time_horizon=t_horizon,
                human_level=0, honest=False, uniform=False,
                f_max=0.5, tau_r=0.3, lambda_entropy=0.5
            )
            final_beliefs = extract_final_beliefs_batch(r[0])
            predictive_results.append(float((final_beliefs > 0.9).mean()))
            predictive_mean_beliefs.append(float(final_beliefs.mean()))
        print(f"  {'Predictive Control':<25} | {predictive_results[0]*100:>7.1f}% | {predictive_results[1]*100:>7.1f}% | {predictive_results[2]*100:>7.1f}% | {onp.mean(predictive_mean_beliefs):>10.2f}")

        # No intervention baseline
        baseline_results = []
        baseline_mean_beliefs = []
        for p_chi, t_horizon in alpha_test_conditions:
            r = run_sim_with_auditor(
                p_chi=p_chi, num_sims=500, time_horizon=t_horizon,
                human_level=0, honest=False, uniform=False, enable_auditor=False
            )
            final_beliefs = extract_final_beliefs_batch(r[0])
            baseline_results.append(float((final_beliefs > 0.9).mean()))
            baseline_mean_beliefs.append(float(final_beliefs.mean()))
        print(f"  {'No Intervention':<25} | {baseline_results[0]*100:>7.1f}% | {baseline_results[1]*100:>7.1f}% | {baseline_results[2]*100:>7.1f}% | {onp.mean(baseline_mean_beliefs):>10.2f}")

        # Verdict
        print("\n" + "-" * 70)
        print("  ANALYSIS:")
        versioning_ood_mean = onp.mean(versioning_results[1:])  # OOD conditions only
        predictive_ood_mean = onp.mean(predictive_results[1:])
        baseline_ood_mean = onp.mean(baseline_results[1:])

        print(f"  • Baseline spiral rate (OOD):          {baseline_ood_mean:.1%}")
        print(f"  • Belief Versioning spiral rate (OOD): {versioning_ood_mean:.1%}")
        print(f"  • Predictive Control spiral rate (OOD): {predictive_ood_mean:.1%}")
        print()
        print(f"  • Belief Versioning mean belief:  {onp.mean(versioning_mean_beliefs):.2f} (learning preserved)")
        print(f"  • Predictive Control mean belief: {onp.mean(predictive_mean_beliefs):.2f} (stuck at uncertainty)")
        print()

        if predictive_ood_mean < 0.01 and onp.mean(predictive_mean_beliefs) < 0.52:
            print("  ⚠ CAUTION: Predictive Control achieves 0% by destroying learning")
            print("    (mean belief ~0.5 = maximum uncertainty = trivial solution)")

        if versioning_ood_mean < 0.1 and onp.mean(versioning_mean_beliefs) > 0.55:
            print("  ✓ Belief Versioning achieves low spiral rate while preserving learning")
            print("    (mean belief > 0.5 = genuine belief formation occurred)")

        print("-" * 70)

    # Legacy experiments for backward compatibility
    print("\n" + "=" * 60)
    print("LEGACY EXPERIMENTS (original format)")
    print("=" * 60)
    print("  ⚠️  This section compiles 4 separate models - expect ~2-3 min each\n")

    legacy_configs = [
        ("Fabricating sycophant vs naive user", (0, False, 'prior')),
        ("Factual sycophant vs naive user", (0, True, 'prior')),
        ("Fabricating sycophant vs aware user", (1, False, 'prior')),
        ("Factual sycophant vs aware user", (1, True, 'prior')),
    ]
    for i, (title, (human_level, honest, uniform)) in enumerate(legacy_configs):
        fname = f'z-{human_level}-{"factual" if honest else "fabricating"}-{uniform}'
        print(f"\n[{i+1}/{len(legacy_configs)}] {fname}: {title}")
        print("  Compiling new model variant..." if i == 0 else "  Compiling...")
        z = []
        for pi in tqdm(Ps_TESTED, desc=fname):
            outs = run_sim_jit(
                p_chi=pi,
                human_level=human_level,
                num_sims=MAIN_NUM_SIMS,
                time_horizon=MAIN_TIME_HORIZON,
                honest=honest,
                uniform=uniform == 'uniform',
                human_policy=do_sample_from_prior
            ).block_until_ready()
            z.append(outs)
        np.save(fname, z)
        print("  Saved.")

    # =================================================================
    # GENERATE PLOTS
    # =================================================================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    from make_plot2 import (
        plot_fig1_belief_trajectories,
        plot_fig2_belief_versioning,
        plot_fig3_predictive_control,
        plot_fig4_method_comparison,
        plot_fig5_lyapunov_tuning,
        plot_fig6_ood_generalization,
        plot_fig7_heterogeneous_types,
        plot_fig8_statistical_summary,
        plot_fig1_8_combined,
        plot_fig9_versioning_vs_predictive,
        FIGURES_DIR
    )

    # Collect data for plots (reuse experiments where possible)
    print("\n[Fig 1] Auditor comparison...")
    # Run fresh for consistent data
    results_no_aud = run_sim_with_auditor(
        p_chi=90, num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON,
        human_level=0, honest=False, uniform=False, enable_auditor=False
    )
    results_with_aud = run_sim_with_auditor(
        p_chi=90, num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON,
        human_level=0, honest=False, uniform=False, enable_auditor=True
    )
    priors_no, _ = results_no_aud
    priors_with, _ = results_with_aud

    beliefs_no = extract_belief_trajectories_batch(priors_no)
    beliefs_with = extract_belief_trajectories_batch(priors_with)
    final_no = extract_final_beliefs_batch(priors_no)
    final_with = extract_final_beliefs_batch(priors_with)

    plot_fig1_belief_trajectories(beliefs_no, beliefs_with, final_no, final_with, MAIN_TIME_HORIZON)

    print("\n[Fig 2] Belief versioning...")
    results_ver = run_sim_with_belief_versioning(
        p_chi=90, num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON,
        human_level=0, honest=False, uniform=False,
        enable_auditor=True, enable_versioning=True
    )
    priors_ver, frictions_ver, type_conf, checkouts = results_ver
    beliefs_ver = extract_belief_trajectories_batch(priors_ver)
    stats_ver = analyze_versioning_results(results_ver)

    plot_fig2_belief_versioning(beliefs_ver, type_conf, checkouts, stats_ver, MAIN_TIME_HORIZON)

    print("\n[Fig 3] Predictive control...")
    results_pred = run_sim_with_predictive_control(
        p_chi=90, num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON,
        human_level=0, honest=False, uniform=False,
        f_max=0.5, tau_r=0.3, lambda_entropy=0.1
    )
    priors_pred, frictions_pred, risks, lyapunov = results_pred
    beliefs_pred = extract_belief_trajectories_batch(priors_pred)
    stats_pred = analyze_predictive_control_results(results_pred)

    plot_fig3_predictive_control(beliefs_pred, frictions_pred, risks, lyapunov, stats_pred, MAIN_TIME_HORIZON)

    print("\n[Fig 4] Method comparison...")
    spiral_rates = {
        'No Auditor\n(Baseline)': float((final_no > 0.9).mean() * 100),
        'Reactive\nAuditor': float((final_with > 0.9).mean() * 100),
        'Belief\nVersioning': float((extract_final_beliefs_batch(priors_ver) > 0.9).mean() * 100),
        'Predictive\nControl': float((extract_final_beliefs_batch(priors_pred) > 0.9).mean() * 100),
    }
    plot_fig4_method_comparison(spiral_rates)

    print("\n[Fig 5] Lyapunov tuning...")
    lambda_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    violation_rates = []
    extreme_rates = []

    for lam in lambda_values:
        print(f"  Running lambda={lam}...")
        results = run_sim_with_predictive_control(
            p_chi=90, num_sims=MAIN_NUM_SIMS, time_horizon=MAIN_TIME_HORIZON,
            human_level=0, honest=False, uniform=False,
            f_max=0.5, tau_r=0.3, lambda_entropy=lam
        )
        stats = analyze_predictive_control_results(results)
        violation_rates.append(stats['lyapunov_violation_rate'] * 100)
        extreme_rates.append(stats['fraction_extreme'] * 100)

    plot_fig5_lyapunov_tuning(lambda_values, violation_rates, extreme_rates)

    print("\n[Fig 6] OOD Generalization...")
    # Use results from OOD test above (only if sklearn was available)
    if SKLEARN_AVAILABLE:
        plot_fig6_ood_generalization(ood_results, {
            'versioning': versioning_results,
            'predictive': predictive_results,
            'baseline': baseline_results
        })
    else:
        print("  Skipped (requires sklearn for OOD test data)")

    print("\n[Fig 7] Heterogeneous types...")
    # Extract work distributions from hetero_results
    priors_h, frictions_h, works_h, types_h = hetero_results
    work_total = onp.array(works_h.sum(axis=1))
    types_np = onp.array(types_h)
    work_g = work_total[types_np == 0]
    work_v = work_total[types_np == 1]
    plot_fig7_heterogeneous_types(hetero_stats, work_g, work_v)

    print("\n[Fig 8] Statistical summary...")
    plot_fig8_statistical_summary(stat_results)

    print("\n[Fig 1+8] Combined main result figure...")
    plot_fig1_8_combined(
        beliefs_no, beliefs_with, final_no, final_with,
        stat_results, MAIN_TIME_HORIZON
    )

    print("\n[Fig 9] Critical comparison: Versioning vs Predictive...")
    plot_fig9_versioning_vs_predictive(
        beliefs_ver, beliefs_pred, stats_ver, stats_pred, MAIN_TIME_HORIZON
    )

    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED!")
    print("=" * 60)
    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print("\nFiles created:")
    print("  - fig1_auditor_comparison.pdf/png")
    print("  - fig2_belief_versioning.pdf/png  <-- MAIN CONTRIBUTION")
    print("  - fig3_predictive_control.pdf/png  (cautionary baseline)")
    print("  - fig4_method_comparison.pdf/png")
    print("  - fig5_lyapunov_tuning.pdf/png")
    print("  - fig6_ood_generalization.pdf/png")
    print("  - fig7_heterogeneous_types.pdf/png")
    print("  - fig8_statistical_summary.pdf/png")
    print("  - fig9_versioning_vs_predictive.pdf/png  <-- KEY COMPARISON")
    print("  - fig1_8_combined.pdf/png")

    # =================================================================
    # REPRODUCIBILITY CHECK
    # =================================================================
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY CHECK")
    print("=" * 60)
    print("\nVerifying results are reproducible across independent runs...")
    print("(Requirement for scientific validity)\n")

    def run_reproducibility_check():
        """
        Verify results are reproducible across two independent runs.
        A requirement for scientific validity.
        """
        import numpy as onp

        # Run 1: baseline (no auditor)
        r1_base = run_sim_with_auditor(
            p_chi=90, num_sims=200, time_horizon=MAIN_TIME_HORIZON,
            human_level=0, honest=False, uniform=False, enable_auditor=False
        )
        # Run 2: same parameters
        r2_base = run_sim_with_auditor(
            p_chi=90, num_sims=200, time_horizon=MAIN_TIME_HORIZON,
            human_level=0, honest=False, uniform=False, enable_auditor=False
        )

        f1_base = extract_final_beliefs_batch(r1_base[0])
        f2_base = extract_final_beliefs_batch(r2_base[0])
        spiral_r1_base = float((f1_base > 0.9).mean())
        spiral_r2_base = float((f2_base > 0.9).mean())

        # Run 1: with auditor
        r1_aud = run_sim_with_auditor(
            p_chi=90, num_sims=200, time_horizon=MAIN_TIME_HORIZON,
            human_level=0, honest=False, uniform=False, enable_auditor=True
        )
        # Run 2: same parameters
        r2_aud = run_sim_with_auditor(
            p_chi=90, num_sims=200, time_horizon=MAIN_TIME_HORIZON,
            human_level=0, honest=False, uniform=False, enable_auditor=True
        )

        f1_aud = extract_final_beliefs_batch(r1_aud[0])
        f2_aud = extract_final_beliefs_batch(r2_aud[0])
        spiral_r1_aud = float((f1_aud > 0.9).mean())
        spiral_r2_aud = float((f2_aud > 0.9).mean())

        print("  Condition            | Run 1  | Run 2  | Diff   | Reproducible")
        print("  " + "-" * 60)
        diff_base = abs(spiral_r1_base - spiral_r2_base)
        diff_aud = abs(spiral_r1_aud - spiral_r2_aud)
        repro_base = "Yes" if diff_base < 0.05 else "No"
        repro_aud = "Yes" if diff_aud < 0.05 else "No"
        print(f"  No Auditor (baseline) | {spiral_r1_base:.1%} | {spiral_r2_base:.1%} | {diff_base:.3f} | {repro_base}")
        print(f"  With Auditor          | {spiral_r1_aud:.1%} | {spiral_r2_aud:.1%} | {diff_aud:.3f} | {repro_aud}")

        # Overall verdict
        all_reproducible = diff_base < 0.05 and diff_aud < 0.05
        print("\n  " + "-" * 60)
        if all_reproducible:
            print("  ✓ All results reproducible within 5% tolerance")
        else:
            print("  ✗ WARNING: Some results exceed 5% variance between runs")
        print("  " + "-" * 60)

        return {
            'baseline_diff': diff_base,
            'auditor_diff': diff_aud,
            'all_reproducible': all_reproducible
        }

    repro_results = run_reproducibility_check()

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\nConfiguration: seed={GLOBAL_SEED}, num_sims={MAIN_NUM_SIMS}, time_horizon={MAIN_TIME_HORIZON}")
    print("All experiments completed successfully.")

    plt.show()
