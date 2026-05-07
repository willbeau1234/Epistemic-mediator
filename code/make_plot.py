"""
Demo: Epistemic Auditor - Visualizing Delusional Spiral Detection & Intervention

This script demonstrates the core contribution:
- Fig 1: Belief trajectories with vs. without the Auditor
- Fig 2: Trigger activation (when V_e and ΔH cross thresholds)
- Fig 3: Epistemic Work (W) comparison between user types
- Fig 4: Legacy comparison plots (original format)

Run: python make_plot.py
"""

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from delusion import *
import statsmodels.api as sm

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@jax.jit
def get_spiralers(z):
    """Detect simulations where belief spiraled to certainty (P(H0) > 0.99)."""
    z_ = z.sum(axis=-1)[..., H.H0]
    z_ = (z_ > 0.99).any(axis=-1)
    return z_


def p_to_stars(p):
    """Statistical significance markers."""
    if p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''


def get_confidence_intervals(counts, props, nobs):
    """Compute confidence intervals for proportions."""
    ci_abs = sm.stats.proportion_confint(counts, nobs)
    return np.abs(props - np.array(ci_abs))


def plot_results(z, bar=True, label='', dodge=0, color='k', ylabel=False, alpha=1, ls='-', stars=True):
    """Plot spiral rates with error bars."""
    z_ = get_spiralers(z)
    nobs = z_.shape[-1]
    counts = z_.sum(axis=-1)
    props = z_.mean(axis=-1)
    plt.errorbar(
        Ps_TESTED / P_MAX + dodge,
        props,
        yerr=get_confidence_intervals(counts, props, nobs),
        capsize=3,
        label=label,
        c=color, alpha=alpha, ls=ls
    )
    if stars:
        plt.axhline(props[0], ls=':', c=color, alpha=0.5)
    plt.xlabel('Rate of sycophantic/hallucinated responses (π)')
    if ylabel:
        plt.ylabel('Rate of catastrophic\ndelusional spiraling')


# =============================================================================
# DEMO 1: QUICK VISUALIZATION (No pre-saved data required)
# =============================================================================

def run_demo():
    """
    Run a quick demo showing the Auditor's effect on belief trajectories.

    This generates Fig. 1 for the paper:
    - Left: Without Auditor → beliefs spiral toward certainty
    - Right: With Auditor → spiral is interrupted
    """
    print("=" * 60)
    print("EPISTEMIC AUDITOR DEMO")
    print("=" * 60)
    print("\nRunning simulations (this may take a moment on first run)...")

    # Parameters
    num_sims = 200
    time_horizon = 50
    p_chi = 90  # 90% sycophantic bot

    # Run WITHOUT auditor
    print("\n[1/2] Simulating WITHOUT Epistemic Auditor...")
    results_no_auditor = run_sim_with_auditor(
        p_chi=p_chi,
        num_sims=num_sims,
        time_horizon=time_horizon,
        human_level=0,
        honest=False,
        uniform=False,
        enable_auditor=False
    )
    priors_no_aud, _ = results_no_auditor
    priors_no_aud = priors_no_aud.block_until_ready()

    # Run WITH auditor
    print("[2/2] Simulating WITH Epistemic Auditor...")
    results_with_auditor = run_sim_with_auditor(
        p_chi=p_chi,
        num_sims=num_sims,
        time_horizon=time_horizon,
        human_level=0,
        honest=False,
        uniform=False,
        enable_auditor=True,
        tau_v=0.01,
        tau_h=-0.02
    )
    priors_with_aud, frictions = results_with_auditor
    priors_with_aud = priors_with_aud.block_until_ready()
    frictions = frictions.block_until_ready()

    # =============================================================================
    # FIGURE 1: Belief Trajectories Comparison
    # =============================================================================
    print("\nGenerating Figure 1: Belief Trajectories...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot WITHOUT auditor
    ax = axes[0]
    for i in range(min(100, num_sims)):
        beliefs = extract_belief_trajectory(priors_no_aud[i])
        ax.plot(range(len(beliefs)), beliefs, alpha=0.2, color='red', linewidth=0.8)

    # Plot mean trajectory
    all_beliefs_no = np.array([extract_belief_trajectory(priors_no_aud[i]) for i in range(num_sims)])
    mean_beliefs_no = all_beliefs_no.mean(axis=0)
    ax.plot(range(len(mean_beliefs_no)), mean_beliefs_no, color='darkred', linewidth=2.5, label='Mean trajectory')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Uncertainty')
    ax.axhline(y=0.9, color='black', linestyle=':', alpha=0.5, label='High certainty')
    ax.set_xlabel('Conversation Turn', fontsize=12)
    ax.set_ylabel('P(H=1) — Certainty', fontsize=12)
    ax.set_title('WITHOUT Auditor: Delusional Spiral', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, time_horizon)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot WITH auditor
    ax = axes[1]
    for i in range(min(100, num_sims)):
        beliefs = extract_belief_trajectory(priors_with_aud[i])
        ax.plot(range(len(beliefs)), beliefs, alpha=0.2, color='blue', linewidth=0.8)

    # Plot mean trajectory
    all_beliefs_with = np.array([extract_belief_trajectory(priors_with_aud[i]) for i in range(num_sims)])
    mean_beliefs_with = all_beliefs_with.mean(axis=0)
    ax.plot(range(len(mean_beliefs_with)), mean_beliefs_with, color='darkblue', linewidth=2.5, label='Mean trajectory')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Uncertainty')
    ax.axhline(y=0.9, color='black', linestyle=':', alpha=0.5, label='High certainty')
    ax.set_xlabel('Conversation Turn', fontsize=12)
    ax.set_ylabel('P(H=1) — Certainty', fontsize=12)
    ax.set_title('WITH Auditor: Spiral Interrupted', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, time_horizon)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. 1: Effect of Epistemic Auditor on Belief Dynamics\n'
                 r'$\mathcal{T} = \mathbb{I}[V_e > \tau_v \wedge \Delta H < \tau_h]$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig1_belief_trajectories.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('fig1_belief_trajectories.png', bbox_inches='tight', dpi=150)
    print("  Saved: fig1_belief_trajectories.pdf/png")

    # =============================================================================
    # FIGURE 2: Sensor Dynamics and Trigger Activation
    # =============================================================================
    print("\nGenerating Figure 2: Sensor Dynamics...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pick a representative simulation
    sim_idx = 0
    beliefs = extract_belief_trajectory(priors_with_aud[sim_idx])
    friction_trace = frictions[sim_idx]

    # Compute V_e and ΔH for this trajectory
    v_e_list = [0.0] * len(beliefs)
    delta_h_list = [0.0] * len(beliefs)

    # Compute entropy at each step
    entropy_trace = []
    for t in range(len(priors_with_aud[sim_idx])):
        prior_t = priors_with_aud[sim_idx][t]
        p_norm = prior_t / (prior_t.sum() + 1e-10)
        p_norm = np.clip(p_norm, 1e-10, 1.0)
        entropy_trace.append(float(-np.sum(p_norm * np.log(p_norm))))
    entropy_trace = np.array(entropy_trace)

    for t in range(3, len(beliefs)):
        v_e_list[t] = float(np.mean(np.diff(beliefs[t-3:t+1])))
        delta_h_list[t] = float(np.mean(np.diff(entropy_trace[t-3:t+1])))

    v_e_trace = np.array(v_e_list)
    delta_h_trace = np.array(delta_h_list)

    # Plot belief trajectory
    ax = axes[0, 0]
    ax.plot(beliefs, color='blue', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Turn')
    ax.set_ylabel('P(H=1)')
    ax.set_title('(A) Belief Trajectory')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Plot entrenchment velocity
    ax = axes[0, 1]
    ax.plot(v_e_trace, color='orange', linewidth=2, label=r'$V_e = dP(H)/dt$')
    ax.axhline(y=0.01, color='red', linestyle='--', label=r'$\tau_v$ threshold')
    ax.set_xlabel('Turn')
    ax.set_ylabel(r'$V_e$')
    ax.set_title('(B) Entrenchment Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot entropy decay
    ax = axes[1, 0]
    ax.plot(delta_h_trace, color='purple', linewidth=2, label=r'$\Delta H$')
    ax.axhline(y=-0.02, color='red', linestyle='--', label=r'$\tau_h$ threshold')
    ax.set_xlabel('Turn')
    ax.set_ylabel(r'$\Delta H$')
    ax.set_title('(C) Entropy Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot friction (interventions)
    ax = axes[1, 1]
    ax.fill_between(range(len(friction_trace)), friction_trace, alpha=0.7, color='green', label='Friction F')
    ax.set_xlabel('Turn')
    ax.set_ylabel('Friction Magnitude')
    ax.set_title(r'(D) Auditor Interventions: $\mathcal{T}$ Activated')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. 2: Sensor Dynamics and Trigger Mechanism', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('fig2_sensor_dynamics.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('fig2_sensor_dynamics.png', bbox_inches='tight', dpi=150)
    print("  Saved: fig2_sensor_dynamics.pdf/png")

    # =============================================================================
    # FIGURE 3: Epistemic Work Comparison
    # =============================================================================
    print("\nGenerating Figure 3: Epistemic Work Distribution...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute epistemic work for all simulations
    work_no_auditor = []
    work_with_auditor = []

    for i in range(num_sims):
        w_no = compute_epistemic_work_trajectory(priors_no_aud[i])
        # Pass friction sequence to exclude intervention timesteps
        # W measures genuine belief updating only, not forced regularization
        w_with = compute_epistemic_work_trajectory(
            priors_with_aud[i],
            friction_sequence=frictions[i]
        )
        work_no_auditor.append(float(w_no[-1]))
        work_with_auditor.append(float(w_with[-1]))

    work_no_auditor = np.array(work_no_auditor)
    work_with_auditor = np.array(work_with_auditor)

    # Histogram
    bins = np.linspace(0, max(work_no_auditor.max(), work_with_auditor.max()), 30)
    ax.hist(work_no_auditor, bins=bins, alpha=0.6, color='red', label='Without Auditor', density=True)
    ax.hist(work_with_auditor, bins=bins, alpha=0.6, color='blue', label='With Auditor', density=True)

    ax.axvline(work_no_auditor.mean(), color='darkred', linestyle='--', linewidth=2,
               label=f'Mean (no aud): {work_no_auditor.mean():.2f}')
    ax.axvline(work_with_auditor.mean(), color='darkblue', linestyle='--', linewidth=2,
               label=f'Mean (with aud): {work_with_auditor.mean():.2f}')

    ax.set_xlabel(r'Total Epistemic Work $W = \sum_t D_{KL}(P_t \| P_{t-1})$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Fig. 3: Distribution of Epistemic Work\n'
                 '(Higher W = more belief updating, indicates Growth-type behavior)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig3_epistemic_work.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('fig3_epistemic_work.png', bbox_inches='tight', dpi=150)
    print("  Saved: fig3_epistemic_work.pdf/png")

    # =============================================================================
    # SUMMARY STATISTICS
    # =============================================================================
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Final beliefs
    final_no = all_beliefs_no[:, -1]
    final_with = all_beliefs_with[:, -1]

    print(f"\nFinal Belief P(H=1):")
    print(f"  Without Auditor: {final_no.mean():.3f} ± {final_no.std():.3f}")
    print(f"  With Auditor:    {final_with.mean():.3f} ± {final_with.std():.3f}")

    print(f"\nSpiral Rate (P(H=1) > 0.9):")
    print(f"  Without Auditor: {(final_no > 0.9).mean()*100:.1f}%")
    print(f"  With Auditor:    {(final_with > 0.9).mean()*100:.1f}%")

    print(f"\nEpistemic Work W:")
    print(f"  Without Auditor: {work_no_auditor.mean():.2f} ± {work_no_auditor.std():.2f}")
    print(f"  With Auditor:    {work_with_auditor.mean():.2f} ± {work_with_auditor.std():.2f}")

    total_interventions = (frictions > 0).sum()
    print(f"\nAuditor Interventions:")
    print(f"  Total triggers: {total_interventions}")
    print(f"  Mean per simulation: {total_interventions / num_sims:.1f}")

    print("\n" + "=" * 60)
    print("Demo complete! Check the generated PDF/PNG files.")
    print("=" * 60)

    # Run statistical validation and threshold ablation
    spiral_grid, p_value = run_statistical_validation(
        num_sims=num_sims,
        time_horizon=time_horizon,
        p_chi=p_chi
    )
    print(f"\nFig 4 generated. Overall p-value: {p_value:.4f}")

    plt.show()


def run_statistical_validation(num_sims=200, time_horizon=50, p_chi=90):
    """
    Statistical validation of epistemic work difference and
    threshold robustness across the tau_v / tau_h parameter grid.

    Generates:
    - Mann-Whitney U test on W distributions
    - Fig 4: Spiral rate heatmap across threshold combinations
    """
    from scipy import stats

    print("\nRunning statistical validation...")

    # Run baseline (no auditor)
    results_no = run_sim_with_auditor(
        p_chi=p_chi, num_sims=num_sims, time_horizon=time_horizon,
        human_level=0, honest=False, uniform=False, enable_auditor=False
    )
    priors_no, _ = results_no
    priors_no = priors_no.block_until_ready()

    # Run with auditor at chosen thresholds
    results_with = run_sim_with_auditor(
        p_chi=p_chi, num_sims=num_sims, time_horizon=time_horizon,
        human_level=0, honest=False, uniform=False,
        enable_auditor=True, tau_v=0.01, tau_h=-0.02
    )
    priors_with, frictions_with = results_with
    priors_with = priors_with.block_until_ready()
    frictions_with = frictions_with.block_until_ready()

    # Compute W with circularity correction
    work_no, work_with = [], []
    for i in range(num_sims):
        work_no.append(float(
            compute_epistemic_work_trajectory(priors_no[i])[-1]
        ))
        work_with.append(float(
            compute_epistemic_work_trajectory(
                priors_with[i],
                friction_sequence=frictions_with[i]
            )[-1]
        ))

    work_no = np.array(work_no)
    work_with = np.array(work_with)

    # Mann-Whitney U test
    stat, p_value = stats.mannwhitneyu(
        work_with, work_no, alternative='greater'
    )
    print(f"\nMann-Whitney U Test (W_auditor > W_baseline):")
    print(f"  U statistic:  {stat:.1f}")
    print(f"  p-value:      {p_value:.2e}")
    print(f"  -log10(p):    {-np.log10(p_value):.1f}")
    print(f"  Significant:  {'YES' if p_value < 0.05 else 'NO'}")
    print(f"  W baseline:   {work_no.mean():.4f} +/- {work_no.std():.4f}")
    print(f"  W auditor:    {work_with.mean():.4f} +/- {work_with.std():.4f}")

    # Threshold ablation grid
    tau_v_values = [0.005, 0.01, 0.02, 0.05]
    tau_h_values = [-0.01, -0.02, -0.05, -0.10]

    print(f"\nSpiral Rate (%) across threshold grid:")
    print(f"{'':>10}", end="")
    for tau_h in tau_h_values:
        print(f"  tau_h={tau_h:.2f}", end="")
    print()

    spiral_grid = np.zeros((len(tau_v_values), len(tau_h_values)))

    for i, tau_v in enumerate(tau_v_values):
        print(f"tau_v={tau_v:.3f}", end="")
        for j, tau_h in enumerate(tau_h_values):
            results = run_sim_with_auditor(
                p_chi=p_chi, num_sims=num_sims, time_horizon=time_horizon,
                human_level=0, honest=False, uniform=False,
                enable_auditor=True, tau_v=tau_v, tau_h=tau_h
            )
            priors_grid, _ = results
            priors_grid = priors_grid.block_until_ready()
            finals = np.array([
                float(extract_belief_trajectory(priors_grid[k])[-1])
                for k in range(num_sims)
            ])
            spiral_rate = float((finals > 0.9).mean() * 100)
            spiral_grid = spiral_grid.at[i, j].set(spiral_rate)
            print(f"  {spiral_rate:>8.1f}%", end="")
        print()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        spiral_grid, cmap='RdYlGn_r',
        aspect='auto', vmin=0, vmax=55
    )
    ax.set_xticks(range(len(tau_h_values)))
    ax.set_xticklabels([f'{v:.2f}' for v in tau_h_values])
    ax.set_yticks(range(len(tau_v_values)))
    ax.set_yticklabels([f'{v:.3f}' for v in tau_v_values])
    ax.set_xlabel(r'$\tau_h$ (entropy decay threshold)', fontsize=12)
    ax.set_ylabel(r'$\tau_v$ (entrenchment velocity threshold)', fontsize=12)
    ax.set_title(
        'Fig. 4: Spiral Rate (%) across Threshold Grid\n'
        '(Green = effective intervention, Red = ineffective)',
        fontsize=13
    )
    plt.colorbar(im, ax=ax, label='Spiral Rate (%)')

    # Mark chosen thresholds
    chosen_i = tau_v_values.index(0.01)
    chosen_j = tau_h_values.index(-0.02)
    ax.add_patch(plt.Rectangle(
        (chosen_j - 0.5, chosen_i - 0.5), 1, 1,
        fill=False, edgecolor='blue', linewidth=3
    ))
    ax.text(
        chosen_j, chosen_i - 0.6,
        'Chosen', color='blue', ha='center', fontsize=9
    )

    plt.tight_layout()
    plt.savefig('fig4_threshold_ablation.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('fig4_threshold_ablation.png', bbox_inches='tight', dpi=150)
    print("\n  Saved: fig4_threshold_ablation.pdf/png")

    return spiral_grid, p_value


# =============================================================================
# LEGACY PLOTTING (requires pre-generated .npy files)
# =============================================================================

def plot_legacy():
    """Generate the original paper's plots (requires running the full simulation first)."""
    print("\nGenerating legacy plots (requires .npy files from full simulation)...")

    try:
        plt.figure(figsize=(6, 9))
        i, j, n = 4, 1, 0

        plt.subplot(i, j, n := n + 1)
        plt.title('(A) Naive user, hallucinating bot')
        z = np.load('z-0-fabricating-prior.npy')
        plot_results(z, bar=False, color='r', ylabel=True, label='Sycophantic')
        plt.ylim(-0.01, 0.61)

        z = np.load('z-0-fabricating-uniform.npy')
        plot_results(z, bar=False, color='r', stars=False, ls='--', label='Non-sycophantic')
        plt.legend()

        plt.subplot(i, j, n := n + 1)
        plt.title('(B) Naive user, factual bot')
        z = np.load('z-0-factual-prior.npy')
        plot_results(z, bar=False, ylabel=True, color='b')
        plt.ylim(-0.01, 0.61)

        plt.subplot(i, j, n := n + 1)
        plt.title('(C) Informed user, hallucinating bot')
        z = np.load('z-1-fabricating-prior.npy')
        plot_results(z, bar=False, color='g', ylabel=True, label='Sycophantic')
        plt.ylim(-0.001, 0.017)

        z = np.load('z-1-fabricating-uniform.npy')
        plot_results(z, bar=False, color='g', stars=False, ls='--', label='Non-sycophantic')
        plt.legend()

        plt.subplot(i, j, n := n + 1)
        plt.title('(D) Informed user, factual bot')
        z = np.load('z-1-factual-prior.npy')
        plot_results(z, bar=False, ylabel=True, color='m')
        plt.ylim(-0.001, 0.017)

        plt.tight_layout()
        plt.savefig('extensions.pdf')
        print("  Saved: extensions.pdf")

    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("  Run 'python delusion.py' first to generate the .npy data files.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--legacy':
        plot_legacy()
    else:
        run_demo()
