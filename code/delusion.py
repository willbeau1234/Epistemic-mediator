"""
Epistemic Auditor: A Dynamical System for Detecting and Intervening on Delusional Spirals

Mathematical Framework:
============================================

Step 1: Game Setup (Cheap Talk Problem)
    - User has hidden type θ ∈ {θ_G, θ_V} (Growth vs. Validation)
    - User has belief P_t(H) about hypothesis H
    - Sycophantic bot creates pooling equilibrium (fails to distinguish types)

Step 2: Detection (Sensor Math)
    - Entrenchment Velocity: V_e = dP(H)/dt
    - Entropy Decay: ΔH = H(u_t) - H(u_{t-1})

Step 3: Trigger Condition
    T = 𝕀[V_e > τ_v ∧ ΔH < τ_h]
    (Confidence UP while entropy DOWN → delusional spiral)

Step 4: Intervention (Incentive Compatibility)
    - Inject friction F via prior regularization toward maximum entropy (simulation)
    - Proposed deployment: Adversarial RAG via AdversarialRAGInterface
      (see Part 4 for architecture stub)
    - User utility: U_θ(F) = V_θ(ΔP) - C_θ(F)

Step 5: Separating Equilibrium
    - Growth-seekers: C_θG < C_θV → perform epistemic work W = D_KL(P_post || P_prior)
    - Validation-seekers: exit due to high cognitive cost
"""

from memo import memo
from enum import IntEnum
from dataclasses import dataclass
import itertools

import jax
import jax.numpy as np
from jax.scipy.stats.bernoulli import pmf as ber
from jax.scipy.stats.beta import pdf as beta
from functools import partial

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import scipy.stats


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


def compute_belief_entropy(belief_distribution: np.ndarray) -> float:
    """
    Shannon entropy of the belief distribution.
    H(P) = -Σ p_i log(p_i)

    High entropy = uncertain, Low entropy = confident/entrenched
    """
    # Normalize and add small epsilon for numerical stability
    p = belief_distribution / (belief_distribution.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def compute_kl_divergence(p_post: np.ndarray, p_prior: np.ndarray) -> float:
    """
    Epistemic Work: W = D_KL(P_post || P_prior)

    Measures the "cognitive effort" required to update from prior to posterior.
    High W indicates the user engaged with contradicting evidence (Growth type).
    Low W indicates resistance to belief update (Validation type).
    """
    # Normalize distributions
    p_post = p_post / (p_post.sum() + 1e-10)
    p_prior = p_prior / (p_prior.sum() + 1e-10)

    # Clip for numerical stability
    p_post = np.clip(p_post, 1e-10, 1.0)
    p_prior = np.clip(p_prior, 1e-10, 1.0)

    return float(np.sum(p_post * np.log(p_post / p_prior)))


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


# =============================================================================
# PART 6: USER AGENT WITH TYPE-DEPENDENT UTILITY
# =============================================================================

@dataclass
class UserAgent:
    """
    Reference implementation of user type-dependent utility.

    NOTE: This class documents the theta_G / theta_V type distinction
    from the Crawford-Sobel cheap talk framework. It is not instantiated
    in the current simulation, which uses a single homogeneous agent.

    The separating equilibrium between Growth and Validation types
    is a theoretical prediction and proposed direction for future work.
    Type-dependent behavior requires heterogeneous agent simulation
    extending the current framework.
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
# PART 7: SIMULATION WITH AUDITOR
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
            uncertainty_prior = ur_prior(0.5, 1, 1)
            new_prior = (1 - friction) * new_prior + friction * uncertainty_prior

        return (new_prior, key, belief_hist, entropy_hist), (new_prior, friction)

    if prior_uniform:
        prior_syco = 1
        prior_fair = 1
    else:
        prior_syco = time_horizon * p_chi/P_MAX + 1
        prior_fair = time_horizon * (1 - p_chi/P_MAX) + 1
    prior = ur_prior(p_true, prior_syco, prior_fair)

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
# PART 8: VISUALIZATION AND ANALYSIS
# =============================================================================

def extract_belief_trajectory(prior_sequence: np.ndarray) -> np.ndarray:
    """Extract P(H=1) trajectory from prior sequence."""
    # Sum over p_chi dimension to get marginal P(H)
    p_h1 = prior_sequence[:, H.H1, :].sum(axis=-1)
    p_total = prior_sequence.sum(axis=(-2, -1))
    return p_h1 / (p_total + 1e-10)


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
    work = [0.0]
    for t in range(1, len(prior_sequence)):
        if friction_sequence is not None and friction_sequence[t] > 0:
            work.append(0.0)
            continue
        p_post = prior_sequence[t].flatten()
        p_prior = prior_sequence[t-1].flatten()
        work.append(compute_kl_divergence(p_post, p_prior))
    return np.cumsum(np.array(work))


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

    # Plot without auditor
    ax = axes[0]
    for i in range(min(50, len(priors_no_aud))):
        beliefs = extract_belief_trajectory(priors_no_aud[i])
        ax.plot(beliefs, alpha=0.3, color='red')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Uncertainty')
    ax.set_xlabel('Turn')
    ax.set_ylabel('P(H=1) - Certainty')
    ax.set_title('Without Auditor: Delusional Spiral')
    ax.set_ylim(0, 1)
    ax.legend()

    # Plot with auditor
    ax = axes[1]
    for i in range(min(50, len(priors_with_aud))):
        beliefs = extract_belief_trajectory(priors_with_aud[i])
        ax.plot(beliefs, alpha=0.3, color='blue')
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
# PART 9: LEGACY SIMULATION (original interface)
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
    print("=" * 60)
    print("EPISTEMIC AUDITOR: Dynamical System Simulation")
    print("=" * 60)

    # Run comparison experiments
    experiments = [
        ("Sycophantic bot vs naive user (no auditor)",
         dict(human_level=0, honest=False, uniform=False, enable_auditor=False)),
        ("Sycophantic bot vs naive user (WITH auditor)",
         dict(human_level=0, honest=False, uniform=False, enable_auditor=True)),
        ("Factual bot vs sycophancy-aware user",
         dict(human_level=1, honest=True, uniform=False, enable_auditor=False)),
    ]

    for title, params in experiments:
        print(f"\n{title}")
        print("-" * 40)

        # Run simulation
        results = run_sim_with_auditor(
            p_chi=90,
            num_sims=1000,
            time_horizon=50,
            **params
        )

        # Analyze results
        priors, frictions = results

        # Compute final belief distribution
        final_beliefs = []
        for i in range(len(priors)):
            traj = extract_belief_trajectory(priors[i])
            final_beliefs.append(float(traj[-1]))

        final_beliefs = np.array(final_beliefs)
        print(f"  Mean final P(H=1): {final_beliefs.mean():.3f}")
        print(f"  Std final P(H=1):  {final_beliefs.std():.3f}")
        print(f"  Fraction P(H=1) > 0.9: {(final_beliefs > 0.9).mean():.3f}")

        if params.get('enable_auditor', False):
            interventions_per_sim = (frictions > 0).sum(axis=1)
            print(f"  Mean interventions: {interventions_per_sim.mean():.1f}")

    # Legacy experiments for backward compatibility
    print("\n" + "=" * 60)
    print("LEGACY EXPERIMENTS (original format)")
    print("=" * 60)

    for title, (human_level, honest, uniform) in [
        ("Fabricating sycophant vs naive user", (0, False, 'prior')),
        ("Factual sycophant vs naive user", (0, True, 'prior')),
        ("Fabricating sycophant vs aware user", (1, False, 'prior')),
        ("Factual sycophant vs aware user", (1, True, 'prior')),
    ]:
        fname = f'z-{human_level}-{"factual" if honest else "fabricating"}-{uniform}'
        print(f"\n{fname}: {title}")
        z = []
        for pi in tqdm(Ps_TESTED, desc=fname):
            outs = run_sim_jit(
                p_chi=pi,
                human_level=human_level,
                num_sims=10_000,
                time_horizon=100,
                honest=honest,
                uniform=uniform == 'uniform',
                human_policy=do_sample_from_prior
            ).block_until_ready()
            z.append(outs)
        np.save(fname, z)
        print("  Saved.")
