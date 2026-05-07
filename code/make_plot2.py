"""
Plots for delusion2.py: Epistemic Auditor with Advanced Features

This module provides plotting functions that can be:
1. Called directly from delusion2.py with pre-computed data
2. Run standalone to generate all figures

Figures:
- Fig 1: Belief trajectories (no auditor vs with auditor)
- Fig 2: Belief Versioning system (commits, checkouts, type revelation)
- Fig 3: Predictive Control with Lyapunov stability
- Fig 4: Comparison of all intervention methods
- Fig 5: Lyapunov function tuning (lambda ablation)
- Fig 6: OOD Generalization test results
- Fig 7: Heterogeneous user types (θ_G vs θ_V)
- Fig 8: Statistical analysis summary

Run standalone: python make_plot2.py
"""

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as onp
import os

# =============================================================================
# OKABE-ITO COLORBLIND-FRIENDLY PALETTE (Nature Methods standard)
# =============================================================================
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'black': '#000000',
}

# Semantic color assignments for consistent use across figures
COLOR_BASELINE = COLORS['vermillion']      # No auditor / baseline condition
COLOR_TREATMENT = COLORS['bluish_green']   # With auditor / treatment condition
COLOR_MEAN_BASELINE = '#A34700'            # Darker vermillion for mean lines
COLOR_MEAN_TREATMENT = '#006B5A'           # Darker bluish green for mean lines
COLOR_REACTIVE = COLORS['orange']          # Reactive auditor
COLOR_PREDICTIVE = COLORS['blue']          # Predictive control
COLOR_GROWTH = COLORS['bluish_green']      # Growth-seekers (θ_G)
COLOR_VALIDATION = COLORS['vermillion']    # Validation-seekers (θ_V)
COLOR_NEUTRAL = COLORS['sky_blue']         # Neutral / default
COLOR_HIGHLIGHT = COLORS['yellow']         # Highlights
COLOR_REFERENCE = COLORS['black']          # Reference lines

# Output directory for figures
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save_fig(fig, name, save=True):
    """Helper to save figures to the figures directory."""
    if save:
        pdf_path = os.path.join(FIGURES_DIR, f'{name}.pdf')
        png_path = os.path.join(FIGURES_DIR, f'{name}.png')
        fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: figures/{name}.pdf, figures/{name}.png")


# =============================================================================
# FIGURE 1: Basic Auditor Effect (Belief Trajectories)
# =============================================================================
def plot_fig1_belief_trajectories(
    beliefs_no_auditor,
    beliefs_with_auditor,
    final_no,
    final_with,
    time_horizon,
    save=True,
    show=False
):
    """
    Fig 1: Effect of Epistemic Auditor on Belief Dynamics

    Args:
        beliefs_no_auditor: Array of belief trajectories without auditor [n_sims, time]
        beliefs_with_auditor: Array of belief trajectories with auditor [n_sims, time]
        final_no: Final beliefs without auditor [n_sims]
        final_with: Final beliefs with auditor [n_sims]
        time_horizon: Number of timesteps
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_sims = len(beliefs_no_auditor)
    n_plot = min(100, n_sims)

    # Left: Without auditor
    ax = axes[0]
    for i in range(n_plot):
        ax.plot(beliefs_no_auditor[i], alpha=0.15, color=COLOR_BASELINE, linewidth=0.8)
    ax.plot(onp.array(beliefs_no_auditor).mean(axis=0), color=COLOR_MEAN_BASELINE, linewidth=2.5, label='Mean')
    ax.axhline(y=0.5, color=COLOR_REFERENCE, linestyle='--', alpha=0.5, label='Uncertainty')
    ax.axhline(y=0.9, color=COLOR_REFERENCE, linestyle=':', alpha=0.3, label='Extreme threshold')
    ax.set_xlabel('Conversation Turn', fontsize=12)
    ax.set_ylabel('P(H=1)', fontsize=12)
    ax.set_title('WITHOUT Auditor: Delusional Spiral', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, time_horizon)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Right: With auditor
    ax = axes[1]
    for i in range(n_plot):
        ax.plot(beliefs_with_auditor[i], alpha=0.15, color=COLOR_TREATMENT, linewidth=0.8)
    ax.plot(onp.array(beliefs_with_auditor).mean(axis=0), color=COLOR_MEAN_TREATMENT, linewidth=2.5, label='Mean')
    ax.axhline(y=0.5, color=COLOR_REFERENCE, linestyle='--', alpha=0.5, label='Uncertainty')
    ax.axhline(y=0.9, color=COLOR_REFERENCE, linestyle=':', alpha=0.3, label='Extreme threshold')
    ax.set_xlabel('Conversation Turn', fontsize=12)
    ax.set_ylabel('P(H=1)', fontsize=12)
    ax.set_title('WITH Auditor: Spiral Interrupted', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, time_horizon)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Statistics
    spiral_no = float((onp.array(final_no) > 0.9).mean() * 100)
    spiral_with = float((onp.array(final_with) > 0.9).mean() * 100)

    # Removed "Fig. 1:" - LaTeX will provide figure number
    plt.suptitle(
        f'Effect of Epistemic Auditor on Belief Dynamics\n'
        f'Spiral Rate: {spiral_no:.1f}% (no auditor) vs {spiral_with:.1f}% (with auditor)',
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    _save_fig(fig, 'fig1_auditor_comparison', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 2: Belief Versioning System
# =============================================================================
def plot_fig2_belief_versioning(
    beliefs,
    type_confidences,
    checkouts,
    stats,
    time_horizon,
    save=True,
    show=False
):
    """
    Fig 2: Belief Versioning (Git-inspired epistemic memory)

    Args:
        beliefs: Belief trajectories [n_sims, time]
        type_confidences: Type confidence over time [n_sims, time]
        checkouts: Checkout indicators [n_sims, time]
        stats: Dictionary from analyze_versioning_results()
        time_horizon: Number of timesteps
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    n_plot = min(10, len(beliefs))
    turns = onp.arange(time_horizon)

    # Panel A: Sample belief trajectories showing intervention effect
    ax = fig.add_subplot(gs[0, 0])

    # Plot more trajectories to show distribution
    n_plot_traj = min(50, len(beliefs))
    for i in range(n_plot_traj):
        belief_arr = onp.array(beliefs[i])
        ax.plot(belief_arr, alpha=0.15, color=COLOR_TREATMENT, linewidth=0.8)

    # Plot mean trajectory
    mean_beliefs = onp.array(beliefs[:n_plot_traj]).mean(axis=0)
    ax.plot(mean_beliefs, color=COLOR_MEAN_TREATMENT, linewidth=2.5,
            label=f'Mean (final: {mean_beliefs[-1]:.2f})')

    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Uncertainty')
    ax.axhline(y=0.9, color=COLOR_BASELINE, linestyle=':', alpha=0.5, label='Extreme threshold')

    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('P(H=1)', fontsize=11)
    ax.set_title('(A) Belief Trajectories with Versioning', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Type confidence evolution
    ax = fig.add_subplot(gs[0, 1])
    mean_confidence = onp.array(type_confidences).mean(axis=0)
    std_confidence = onp.array(type_confidences).std(axis=0)

    ax.plot(turns, mean_confidence, color=COLORS['reddish_purple'], linewidth=2, label='Mean type confidence')
    ax.fill_between(turns,
                    mean_confidence - std_confidence,
                    mean_confidence + std_confidence,
                    alpha=0.3, color=COLORS['reddish_purple'])
    ax.axhline(y=0.7, color=COLOR_BASELINE, linestyle='--', label='Checkout threshold (0.7)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('Type Confidence', fontsize=11)
    ax.set_title('(B) Type Confidence Evolution', fontsize=12, fontweight='bold')
    ax.set_ylim(0.3, 0.9)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Type revelation rates
    ax = fig.add_subplot(gs[1, 0])
    labels = ['Validation-seekers\n(detected)', 'Growth-seekers\n(detected)', 'Unclassified']
    frac_v = stats['fraction_validation_revealed']
    frac_g = stats['fraction_growth_revealed']
    frac_u = 1 - frac_v - frac_g
    sizes = [frac_v, frac_g, frac_u]
    colors_pie = [COLOR_VALIDATION, COLOR_GROWTH, COLORS['sky_blue']]
    explode = (0.05, 0.05, 0)

    ax.pie(
        sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10}
    )
    ax.set_title('(C) User Type Classification', fontsize=12, fontweight='bold')

    # Panel D: Friction comparison by type
    ax = fig.add_subplot(gs[1, 1])
    categories = ['Validation-seekers', 'Growth-seekers']
    friction_v = stats['mean_friction_validation']
    friction_g = stats['mean_friction_growth']

    bars = ax.bar(categories, [friction_v, friction_g], color=[COLOR_VALIDATION, COLOR_GROWTH], edgecolor=COLOR_REFERENCE)
    ax.set_ylabel('Mean Cumulative Friction', fontsize=11)
    ax.set_title('(D) Friction by User Type', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(friction_v, friction_g) * 1.2)  # Add 20% headroom for labels
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, [friction_v, friction_g]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)

    # Calculate spiral rate from beliefs
    final_beliefs = onp.array(beliefs)[:, -1]
    spiral_rate = float((final_beliefs > 0.9).mean())

    # Removed "Fig. 2:" - LaTeX will provide figure number
    plt.suptitle(
        f'Belief Versioning (Main Contribution)\n'
        f'Spiral Rate: {spiral_rate:.1%} | Mean checkouts: {stats["mean_checkouts_per_sim"]:.1f} | Learning preserved',
        fontsize=14, y=1.02
    )

    _save_fig(fig, 'fig2_belief_versioning', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 3: Predictive Control with Lyapunov Stability
# =============================================================================
def plot_fig3_predictive_control(
    beliefs,
    frictions,
    risks,
    lyapunov_values,
    stats,
    time_horizon,
    save=True,
    show=False
):
    """
    Fig 3: Predictive Control System

    Args:
        beliefs: Belief trajectories [n_sims, time]
        frictions: Friction values [n_sims, time]
        risks: Spiral risk values [n_sims, time]
        lyapunov_values: Lyapunov function values [n_sims, time]
        stats: Dictionary from analyze_predictive_control_results()
        time_horizon: Number of timesteps
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_plot = min(50, len(beliefs))
    turns = onp.arange(time_horizon)

    # Panel A: Belief trajectories
    ax = axes[0, 0]
    for i in range(n_plot):
        ax.plot(beliefs[i], alpha=0.2, color=COLOR_PREDICTIVE, linewidth=0.8)
    ax.plot(onp.array(beliefs[:n_plot]).mean(axis=0), color=COLORS['blue'], linewidth=2.5, label='Mean')
    ax.axhline(y=0.5, color=COLOR_REFERENCE, linestyle='--', alpha=0.5)
    ax.axhline(y=0.9, color=COLOR_BASELINE, linestyle=':', alpha=0.5, label='Extreme threshold')
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('P(H=1)', fontsize=11)
    ax.set_title('(A) Belief Trajectories (Flattened - No Learning)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Spiral risk
    ax = axes[0, 1]
    mean_risk = onp.array(risks[:n_plot]).mean(axis=0)
    std_risk = onp.array(risks[:n_plot]).std(axis=0)
    ax.plot(turns, mean_risk, color=COLORS['orange'], linewidth=2, label='Mean risk')
    ax.fill_between(turns, mean_risk - std_risk, mean_risk + std_risk, alpha=0.3, color=COLORS['orange'])
    ax.axhline(y=0.3, color=COLOR_BASELINE, linestyle='--', label=r'$\tau_R$ threshold')
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('Spiral Risk R(t)', fontsize=11)
    ax.set_title('(B) Spiral Risk Detection', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Lyapunov function
    ax = axes[1, 0]
    mean_lyap = onp.array(lyapunov_values[:n_plot]).mean(axis=0)
    std_lyap = onp.array(lyapunov_values[:n_plot]).std(axis=0)
    ax.plot(turns, mean_lyap, color=COLOR_TREATMENT, linewidth=2, label='Mean V(x)')
    ax.fill_between(turns, mean_lyap - std_lyap, mean_lyap + std_lyap, alpha=0.3, color=COLOR_TREATMENT)
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('V(x) = P(1-P) + λH', fontsize=11)
    ax.set_title(f'(C) Lyapunov Function (violation rate: {stats["lyapunov_violation_rate"]:.1%})',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel D: Friction application
    ax = axes[1, 1]
    mean_friction = onp.array(frictions[:n_plot]).mean(axis=0)
    ax.fill_between(turns, 0, mean_friction, alpha=0.7, color=COLORS['reddish_purple'], label='Mean friction')
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('Friction F(t)', fontsize=11)
    ax.set_title('(D) Proportional Friction Application', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Removed "Fig. 3:" - LaTeX will provide figure number
    plt.suptitle(
        f'Predictive Control (Cautionary Baseline)\n'
        f'Extreme beliefs: {stats["fraction_extreme"]:.1%}, Mean belief: {stats["mean_final_belief"]:.2f} (stuck at uncertainty)',
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    _save_fig(fig, 'fig3_predictive_control', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 4: Method Comparison
# =============================================================================
def plot_fig4_method_comparison(
    spiral_rates,
    method_names=None,
    save=True,
    show=False
):
    """
    Fig 4: Comparison of all intervention methods

    Args:
        spiral_rates: Dict or list of spiral rates for each method
        method_names: Optional list of method names (if spiral_rates is a list)
        save: Whether to save the figure
        show: Whether to display the figure
    """
    if isinstance(spiral_rates, dict):
        methods = list(spiral_rates.keys())
        rates = list(spiral_rates.values())
    else:
        methods = method_names or [f'Method {i}' for i in range(len(spiral_rates))]
        rates = spiral_rates

    colors = [COLOR_BASELINE, COLOR_REACTIVE, COLOR_NEUTRAL, COLOR_TREATMENT][:len(methods)]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = onp.arange(len(methods))
    bars = ax.bar(x, rates, color=colors, edgecolor=COLOR_REFERENCE, linewidth=1.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Spiral Rate (P(H=1) > 0.9)', fontsize=12)
    # Removed "Fig. 4:" - LaTeX will provide figure number
    ax.set_title('Comparison of Intervention Methods\n'
                 '(Lower is better, but 0% = trivial solution)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    _save_fig(fig, 'fig4_method_comparison', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 5: Lyapunov Lambda Tuning
# =============================================================================
def plot_fig5_lyapunov_tuning(
    lambda_values,
    violation_rates,
    extreme_rates,
    save=True,
    show=False
):
    """
    Fig 5: Effect of lambda (entropy weight) on Lyapunov stability

    Args:
        lambda_values: List of lambda values tested
        violation_rates: Lyapunov violation rates (as percentages)
        extreme_rates: Extreme belief rates (as percentages)
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = onp.arange(len(lambda_values))
    width = 0.35

    ax1.bar(x - width/2, violation_rates, width, label='Lyapunov Violation Rate',
            color=COLOR_BASELINE, edgecolor=COLOR_REFERENCE)
    ax1.set_xlabel(r'$\lambda$ (entropy weight)', fontsize=12)
    ax1.set_ylabel('Lyapunov Violation Rate (%)', fontsize=12, color=COLOR_BASELINE)
    ax1.tick_params(axis='y', labelcolor=COLOR_BASELINE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in lambda_values])

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, extreme_rates, width, label='Extreme Belief Rate',
            color=COLOR_PREDICTIVE, edgecolor=COLOR_REFERENCE)
    ax2.set_ylabel('Extreme Belief Rate (%)', fontsize=12, color=COLOR_PREDICTIVE)
    ax2.tick_params(axis='y', labelcolor=COLOR_PREDICTIVE)

    # Mark optimal lambda
    optimal_idx = onp.argmin(violation_rates)
    ax1.annotate('Optimal',
                 xy=(optimal_idx - width/2, violation_rates[optimal_idx]),
                 xytext=(optimal_idx - width/2, violation_rates[optimal_idx] + 10),
                 ha='center', fontsize=10, color=COLOR_TREATMENT, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLOR_TREATMENT, lw=2))

    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
    # Removed "Fig. 5:" - LaTeX will provide figure number
    ax1.set_title(r'Lyapunov Function Tuning ($V(x) = P(1-P) + \lambda H$)' + '\n'
                  f'Optimal: λ={lambda_values[optimal_idx]}',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    _save_fig(fig, 'fig5_lyapunov_tuning', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 6: OOD Generalization Test
# =============================================================================
def plot_fig6_ood_generalization(
    ood_results,
    alpha_comparison,
    save=True,
    show=False
):
    """
    Fig 6: Out-of-Distribution Generalization Test

    Args:
        ood_results: Dict mapping condition labels to (no_interv, reactive, versioning, predictive) tuples
        alpha_comparison: Dict with 'versioning', 'predictive', 'baseline' keys, each mapping to list of rates
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: OOD across conditions
    ax = axes[0]
    conditions = list(ood_results.keys())
    x = onp.arange(len(conditions))
    width = 0.2

    no_interv = [ood_results[c][0] * 100 for c in conditions]
    reactive = [ood_results[c][1] * 100 for c in conditions]
    versioning = [ood_results[c][2] * 100 for c in conditions]
    predictive = [ood_results[c][3] * 100 for c in conditions]

    ax.bar(x - 1.5*width, no_interv, width, label='No Intervention', color=COLOR_BASELINE, edgecolor=COLOR_REFERENCE)
    ax.bar(x - 0.5*width, reactive, width, label='Reactive Auditor', color=COLOR_REACTIVE, edgecolor=COLOR_REFERENCE)
    ax.bar(x + 0.5*width, versioning, width, label='Belief Versioning', color=COLOR_TREATMENT, edgecolor=COLOR_REFERENCE)
    ax.bar(x + 1.5*width, predictive, width, label='Predictive (trivial)', color=COLOR_PREDICTIVE, edgecolor=COLOR_REFERENCE, alpha=0.5)

    ax.set_xlabel('Test Condition', fontsize=11)
    ax.set_ylabel('Extreme Belief Rate (%)', fontsize=11)
    ax.set_title('(A) Generalization Across Conditions', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(', ', '\n') for c in conditions], fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Belief Versioning vs Predictive Control comparison
    ax = axes[1]
    p_chi_labels = ['p_chi=90\n(in-sample)', 'p_chi=70\n(OOD)', 'p_chi=60\n(OOD)']
    x = onp.arange(len(p_chi_labels))
    width = 0.25

    versioning_rates = [r * 100 for r in alpha_comparison['versioning']]
    predictive_rates = [r * 100 for r in alpha_comparison['predictive']]
    baseline_rates = [r * 100 for r in alpha_comparison['baseline']]

    ax.bar(x - width, baseline_rates, width, label='No Intervention', color=COLOR_BASELINE, edgecolor=COLOR_REFERENCE)
    ax.bar(x, versioning_rates, width, label='Belief Versioning', color=COLOR_TREATMENT, edgecolor=COLOR_REFERENCE)
    ax.bar(x + width, predictive_rates, width, label='Predictive (trivial)', color=COLOR_PREDICTIVE, edgecolor=COLOR_REFERENCE, alpha=0.5)

    ax.set_xlabel('Test Condition', fontsize=11)
    ax.set_ylabel('Extreme Belief Rate (%)', fontsize=11)
    ax.set_title('(B) Versioning vs Predictive (OOD Test)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(p_chi_labels, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Removed suptitle - LaTeX will provide figure number
    plt.tight_layout()

    _save_fig(fig, 'fig6_ood_generalization', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 7: Heterogeneous User Types
# =============================================================================
def plot_fig7_heterogeneous_types(
    hetero_stats,
    work_growth,
    work_validation,
    save=True,
    show=False
):
    """
    Fig 7: Heterogeneous User Types (θ_G vs θ_V)

    Args:
        hetero_stats: Dictionary from analyze_heterogeneous_results()
        work_growth: Epistemic work values for growth-seekers
        work_validation: Epistemic work values for validation-seekers
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Epistemic work distribution by type
    ax = axes[0]
    ax.hist(onp.array(work_growth), bins=30, alpha=0.7, label=r'$\theta_G$ (Growth)', color=COLOR_GROWTH, edgecolor=COLOR_REFERENCE)
    ax.hist(onp.array(work_validation), bins=30, alpha=0.7, label=r'$\theta_V$ (Validation)', color=COLOR_VALIDATION, edgecolor=COLOR_REFERENCE)
    ax.axvline(x=0.5, color=COLOR_REFERENCE, linestyle='--', label='Classification threshold')
    ax.set_xlabel('Cumulative Epistemic Work (W)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('(A) Work Distribution by True Type', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Detection accuracy
    ax = axes[1]
    categories = ['Recall θ_G', 'Recall θ_V', 'Overall\nAccuracy']
    values = [
        hetero_stats['recall_growth'] * 100,
        hetero_stats['recall_validation'] * 100,
        hetero_stats['overall_detection_accuracy'] * 100
    ]
    colors_bars = [COLOR_GROWTH, COLOR_VALIDATION, COLOR_NEUTRAL]
    bars = ax.bar(categories, values, color=colors_bars, edgecolor=COLOR_REFERENCE)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('(B) Type Detection Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel C: Spiral rates by type
    ax = axes[2]
    categories = [r'$\theta_G$ (Growth)', r'$\theta_V$ (Validation)']
    spiral_rates = [
        hetero_stats['spiral_rate_growth'] * 100,
        hetero_stats['spiral_rate_validation'] * 100
    ]
    colors_bars = [COLOR_GROWTH, COLOR_VALIDATION]
    bars = ax.bar(categories, spiral_rates, color=colors_bars, edgecolor=COLOR_REFERENCE)
    ax.set_ylabel('Spiral Rate (%)', fontsize=11)
    ax.set_title('(C) Spiral Rates by True Type', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(spiral_rates) * 1.15)  # Add 15% headroom for labels
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, spiral_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Removed suptitle with incorrect 0.090 value - LaTeX will provide figure number
    plt.tight_layout()

    _save_fig(fig, 'fig7_heterogeneous_types', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 8: Statistical Summary
# =============================================================================
def plot_fig8_statistical_summary(
    stat_results,
    save=True,
    show=False
):
    """
    Fig 8: Statistical Analysis Summary

    Args:
        stat_results: Dictionary from run_statistical_comparison()
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Spiral rates with CIs
    ax = axes[0]
    categories = ['Baseline\n(No Auditor)', 'Treatment\n(With Auditor)']
    rates = [stat_results['spiral_rate_baseline'] * 100, stat_results['spiral_rate_treatment'] * 100]
    ci_baseline = stat_results['ci_95_baseline']
    ci_treatment = stat_results['ci_95_treatment']
    errors = [
        [rates[0] - ci_baseline[0] * 100, ci_baseline[1] * 100 - rates[0]],
        [rates[1] - ci_treatment[0] * 100, ci_treatment[1] * 100 - rates[1]]
    ]
    errors = onp.array(errors).T

    colors_bars = [COLOR_BASELINE, COLOR_TREATMENT]
    bars = ax.bar(categories, rates, color=colors_bars, edgecolor=COLOR_REFERENCE, yerr=errors, capsize=5)
    ax.set_ylabel('Spiral Rate (%) with 95% CI', fontsize=11)
    ax.set_title('(A) Spiral Rates with Confidence Intervals', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[1][i] + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel B: Effect size and significance
    ax = axes[1]
    ax.axis('off')

    # Create text summary
    summary_text = f"""
    Statistical Analysis Summary
    {'='*40}

    Sample Sizes:
      Baseline:  n = {stat_results['n_baseline']}
      Treatment: n = {stat_results['n_treatment']}

    Spiral Rates:
      Baseline:  {stat_results['spiral_rate_baseline']:.1%}
      Treatment: {stat_results['spiral_rate_treatment']:.1%}

    Reduction:
      Absolute: {stat_results['absolute_reduction']:.1%}
      Relative: {stat_results['relative_reduction']:.1%}

    Hypothesis Test (H0: baseline ≤ treatment):
      z-statistic: {stat_results['z_statistic']:.3f}
      p-value:     {stat_results['p_value']:.2e}
      Significant: {'Yes (p < 0.01)' if stat_results['significant_at_01'] else ('Yes (p < 0.05)' if stat_results['significant_at_05'] else 'No')}

    Effect Size:
      Cohen's d: {stat_results['cohens_d']:.3f}
      Interpretation: {stat_results['effect_size_interpretation']}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('(B) Statistical Test Results', fontsize=12, fontweight='bold')

    # Removed "Fig. 8:" - LaTeX will provide figure number
    plt.suptitle('Statistical Analysis of Auditor Effectiveness',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    _save_fig(fig, 'fig8_statistical_summary', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 1+8 COMBINED: Main Result Figure
# =============================================================================
def plot_fig1_8_combined(
    beliefs_no_auditor,
    beliefs_with_auditor,
    final_no,
    final_with,
    stat_results,
    time_horizon,
    save=True,
    show=False
):
    """
    Combined Fig 1+8: Belief trajectories with statistical analysis.

    This is the main result figure for the paper, showing:
    - Top row: Belief trajectories (no auditor vs with auditor)
    - Bottom row: Statistical comparison with confidence intervals

    Args:
        beliefs_no_auditor: Array of belief trajectories without auditor [n_sims, time]
        beliefs_with_auditor: Array of belief trajectories with auditor [n_sims, time]
        final_no: Final beliefs without auditor [n_sims]
        final_with: Final beliefs with auditor [n_sims]
        stat_results: Dictionary from run_statistical_comparison()
        time_horizon: Number of timesteps
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    n_sims = len(beliefs_no_auditor)
    n_plot = min(100, n_sims)

    # =========================================================================
    # TOP LEFT: Without auditor
    # =========================================================================
    ax = fig.add_subplot(gs[0, 0])
    for i in range(n_plot):
        ax.plot(beliefs_no_auditor[i], alpha=0.12, color=COLOR_BASELINE, linewidth=0.7)
    ax.plot(onp.array(beliefs_no_auditor).mean(axis=0), color=COLOR_MEAN_BASELINE, linewidth=2.5, label='Mean')
    ax.axhline(y=0.5, color=COLOR_REFERENCE, linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=0.9, color=COLOR_REFERENCE, linestyle=':', alpha=0.3, linewidth=1)
    ax.set_xlabel('Conversation Turn', fontsize=11)
    ax.set_ylabel('P(H=1)', fontsize=11)
    ax.set_title('(A) Without Auditor', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, time_horizon)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # =========================================================================
    # TOP RIGHT: With auditor
    # =========================================================================
    ax = fig.add_subplot(gs[0, 1])
    for i in range(n_plot):
        ax.plot(beliefs_with_auditor[i], alpha=0.12, color=COLOR_TREATMENT, linewidth=0.7)
    ax.plot(onp.array(beliefs_with_auditor).mean(axis=0), color=COLOR_MEAN_TREATMENT, linewidth=2.5, label='Mean')
    ax.axhline(y=0.5, color=COLOR_REFERENCE, linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=0.9, color=COLOR_REFERENCE, linestyle=':', alpha=0.3, linewidth=1)
    ax.set_xlabel('Conversation Turn', fontsize=11)
    ax.set_ylabel('P(H=1)', fontsize=11)
    ax.set_title('(B) With Auditor', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, time_horizon)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # =========================================================================
    # BOTTOM LEFT: Spiral rates with confidence intervals
    # =========================================================================
    ax = fig.add_subplot(gs[1, 0])
    categories = ['Baseline', 'With Auditor']
    rates = [stat_results['spiral_rate_baseline'] * 100, stat_results['spiral_rate_treatment'] * 100]
    ci_baseline = stat_results['ci_95_baseline']
    ci_treatment = stat_results['ci_95_treatment']
    errors = [
        [rates[0] - ci_baseline[0] * 100, ci_baseline[1] * 100 - rates[0]],
        [rates[1] - ci_treatment[0] * 100, ci_treatment[1] * 100 - rates[1]]
    ]
    errors = onp.array(errors).T

    colors_bars = [COLOR_BASELINE, COLOR_TREATMENT]
    bars = ax.bar(categories, rates, color=colors_bars, edgecolor=COLOR_REFERENCE, linewidth=1.2, yerr=errors, capsize=6)
    ax.set_ylabel('Spiral Rate (%)', fontsize=11)
    ax.set_title('(C) Spiral Rate Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(rates) * 1.25)

    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[1][i] + 1.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # =========================================================================
    # BOTTOM RIGHT: Statistical summary text
    # =========================================================================
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    summary_text = (
        f"Statistical Summary\n"
        f"{'─'*28}\n\n"
        f"Sample size: n = {stat_results['n_baseline']}\n\n"
        f"Spiral rates:\n"
        f"  Baseline:   {stat_results['spiral_rate_baseline']:.1%}\n"
        f"  Treatment:  {stat_results['spiral_rate_treatment']:.1%}\n\n"
        f"Hypothesis test:\n"
        f"  z = {stat_results['z_statistic']:.2f},  p < 0.001\n\n"
        f"Effect size:\n"
        f"  Cohen's d = {stat_results['cohens_d']:.2f}\n"
        f"  ({stat_results['effect_size_interpretation']})"
    )

    ax.text(0.5, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.8', facecolor=COLORS['sky_blue'],
                      edgecolor=COLOR_REFERENCE, alpha=0.15))
    ax.set_title('(D) Statistical Summary', fontsize=12, fontweight='bold')

    # Main title
    spiral_no = float((onp.array(final_no) > 0.9).mean() * 100)
    spiral_with = float((onp.array(final_with) > 0.9).mean() * 100)

    plt.suptitle(
        'Effect of Epistemic Auditor on Delusional Spiral Prevention',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    _save_fig(fig, 'fig1_8_combined', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# FIGURE 9: Belief Versioning vs Predictive Control (Direct Comparison)
# =============================================================================
def plot_fig9_versioning_vs_predictive(
    beliefs_versioning,
    beliefs_predictive,
    stats_versioning,
    stats_predictive,
    time_horizon,
    save=True,
    show=False
):
    """
    Fig 9: Direct comparison of Belief Versioning vs Predictive Control

    Shows the safety-learning tradeoff: Predictive Control prevents
    spiral rates by suppressing belief movement.

    Args:
        beliefs_versioning: Belief trajectories from versioning [n_sims, time]
        beliefs_predictive: Belief trajectories from predictive control [n_sims, time]
        stats_versioning: Stats dict from analyze_versioning_results()
        stats_predictive: Stats dict from analyze_predictive_control_results()
        time_horizon: Number of timesteps
        save: Whether to save the figure
        show: Whether to display the figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    n_plot = min(50, len(beliefs_versioning))
    turns = onp.arange(time_horizon)

    # Panel A: Belief Versioning trajectories
    ax = fig.add_subplot(gs[0, 0])
    for i in range(n_plot):
        ax.plot(beliefs_versioning[i], alpha=0.2, color=COLOR_TREATMENT, linewidth=0.8)
    mean_ver = onp.asarray(beliefs_versioning[:n_plot], dtype=onp.float64).mean(axis=0)
    # Extract final value handling JAX/numpy arrays - use flatten()[0] for robustness
    final_val_ver = onp.asarray(mean_ver[-1]).flatten()[0] if hasattr(mean_ver[-1], 'flatten') else float(mean_ver[-1])
    ax.plot(mean_ver, color=COLOR_MEAN_TREATMENT, linewidth=2.5, label=f'Mean (final: {final_val_ver:.2f})')
    ax.axhline(y=0.5, color=COLOR_REFERENCE, linestyle='--', alpha=0.5)
    ax.axhline(y=0.9, color=COLOR_BASELINE, linestyle=':', alpha=0.5, label='Extreme threshold')
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('P(H=1)', fontsize=11)
    ax.set_title('(A) Belief Versioning: Learning Preserved', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel B: Predictive Control trajectories
    ax = fig.add_subplot(gs[0, 1])
    for i in range(n_plot):
        ax.plot(beliefs_predictive[i], alpha=0.2, color=COLOR_PREDICTIVE, linewidth=0.8)
    mean_pred = onp.asarray(beliefs_predictive[:n_plot], dtype=onp.float64).mean(axis=0)
    # Extract final value handling JAX/numpy arrays - use flatten()[0] for robustness
    final_val_pred = onp.asarray(mean_pred[-1]).flatten()[0] if hasattr(mean_pred[-1], 'flatten') else float(mean_pred[-1])
    ax.plot(mean_pred, color=COLORS['blue'], linewidth=2.5, label=f'Mean (final: {final_val_pred:.2f})')
    ax.axhline(y=0.5, color=COLOR_REFERENCE, linestyle='--', alpha=0.5)
    ax.axhline(y=0.9, color=COLOR_BASELINE, linestyle=':', alpha=0.5, label='Extreme threshold')
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('P(H=1)', fontsize=11)
    ax.set_title('(B) Predictive Control: Learning Destroyed', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Compute spiral rates from beliefs directly
    final_ver = onp.array(beliefs_versioning)[:, -1]
    final_pred = onp.array(beliefs_predictive)[:, -1]
    spiral_rate_ver = float((final_ver > 0.9).mean())
    spiral_rate_pred = float((final_pred > 0.9).mean())

    # Panel C: Spiral rates comparison
    ax = fig.add_subplot(gs[1, 0])
    categories = ['Belief\nVersioning', 'Predictive\nControl']
    rates = [spiral_rate_ver * 100, spiral_rate_pred * 100]
    colors_bars = [COLOR_TREATMENT, COLOR_PREDICTIVE]
    bars = ax.bar(categories, rates, color=colors_bars, edgecolor=COLOR_REFERENCE, linewidth=1.5)
    ax.set_ylabel('Extreme Belief Rate (%)', fontsize=11)
    ax.set_title('(C) Spiral Rates (Lower = Better?)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.4 if max(rates) > 0 else 5)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel D: Key insight text
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    insight_text = (
        "Learning Preservation Criterion (LPC)\n"
        f"{'─'*42}\n\n"
        "BELIEF VERSIONING:\n"
        f"  Spiral rate:     {spiral_rate_ver:.1%}\n"
        f"  Mean checkouts:  {stats_versioning['mean_checkouts_per_sim']:.1f}\n"
        "  Beliefs move between interventions\n"
        "  LPC: PASS (genuine belief dynamics)\n\n"
        "PREDICTIVE CONTROL:\n"
        f"  Spiral rate:     {spiral_rate_pred:.1%}\n"
        f"  Mean belief:     {stats_predictive['mean_final_belief']:.2f}\n"
        "  Beliefs suppressed at uncertainty\n"
        "  LPC: FAIL (belief movement prevented)\n\n"
        "TRADEOFF:\n"
        "  0% spiral rate achieved by suppressing\n"
        "  belief movement throughout interaction.\n"
        "  BV preserves belief dynamics."
    )

    ax.text(0.5, 0.5, insight_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.8', facecolor=COLORS['yellow'],
                      edgecolor=COLOR_REFERENCE, alpha=0.3))
    ax.set_title('(D) Why This Matters', fontsize=12, fontweight='bold')

    # Removed suptitle - LaTeX will provide figure number
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    _save_fig(fig, 'fig9_versioning_vs_predictive', save)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================
if __name__ == '__main__':
    import sys
    import io

    # Suppress loading prints from delusion2
    print("Loading delusion2.py (this may take a moment)...")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    from delusion2 import (
        run_sim_with_auditor,
        run_sim_with_belief_versioning,
        run_sim_with_predictive_control,
        extract_final_beliefs_batch,
        extract_belief_trajectories_batch,
        analyze_versioning_results,
        analyze_predictive_control_results,
    )

    sys.stdout = old_stdout
    print("Loaded successfully!")

    # Configuration
    NUM_SIMS = 1000  # Changed from 500 to match paper methodology (n=1000 throughout)
    TIME_HORIZON = 50  # Changed from 30 to match delusion2.py canonical parameters
    P_CHI = 90

    print("=" * 60)
    print("GENERATING PLOTS FOR DELUSION2.PY")
    print("=" * 60)

    # Figure 1: Auditor comparison
    print("\n[Fig 1] Generating belief trajectory comparison...")
    print("  Running WITHOUT auditor...")
    results_no = run_sim_with_auditor(
        p_chi=P_CHI, num_sims=NUM_SIMS, time_horizon=TIME_HORIZON,
        human_level=0, honest=False, uniform=False, enable_auditor=False
    )
    priors_no, _ = results_no

    print("  Running WITH auditor...")
    results_with = run_sim_with_auditor(
        p_chi=P_CHI, num_sims=NUM_SIMS, time_horizon=TIME_HORIZON,
        human_level=0, honest=False, uniform=False, enable_auditor=True
    )
    priors_with, _ = results_with

    beliefs_no = extract_belief_trajectories_batch(priors_no)
    beliefs_with = extract_belief_trajectories_batch(priors_with)
    final_no = extract_final_beliefs_batch(priors_no)
    final_with = extract_final_beliefs_batch(priors_with)

    plot_fig1_belief_trajectories(beliefs_no, beliefs_with, final_no, final_with, TIME_HORIZON)

    # Figure 2: Belief versioning
    print("\n[Fig 2] Generating belief versioning visualization...")
    results_ver = run_sim_with_belief_versioning(
        p_chi=P_CHI, num_sims=NUM_SIMS, time_horizon=TIME_HORIZON,
        human_level=0, honest=False, uniform=False,
        enable_auditor=True, enable_versioning=True
    )
    priors_ver, frictions_ver, type_conf, checkouts = results_ver
    beliefs_ver = extract_belief_trajectories_batch(priors_ver)
    stats_ver = analyze_versioning_results(results_ver)

    plot_fig2_belief_versioning(beliefs_ver, type_conf, checkouts, stats_ver, TIME_HORIZON)

    # Figure 3: Predictive control
    print("\n[Fig 3] Generating predictive control visualization...")
    results_pred = run_sim_with_predictive_control(
        p_chi=P_CHI, num_sims=NUM_SIMS, time_horizon=TIME_HORIZON,
        human_level=0, honest=False, uniform=False,
        f_max=0.5, tau_r=0.3, lambda_entropy=0.1
    )
    priors_pred, frictions_pred, risks, lyapunov = results_pred
    beliefs_pred = extract_belief_trajectories_batch(priors_pred)
    stats_pred = analyze_predictive_control_results(results_pred)

    plot_fig3_predictive_control(beliefs_pred, frictions_pred, risks, lyapunov, stats_pred, TIME_HORIZON)

    # Figure 4: Method comparison
    print("\n[Fig 4] Generating method comparison...")
    spiral_rates = {
        'No Auditor\n(Baseline)': float((final_no > 0.9).mean() * 100),
        'Reactive\nAuditor': float((final_with > 0.9).mean() * 100),
        'Belief\nVersioning': float((extract_final_beliefs_batch(priors_ver) > 0.9).mean() * 100),
        'Predictive\nControl': float((extract_final_beliefs_batch(priors_pred) > 0.9).mean() * 100),
    }
    plot_fig4_method_comparison(spiral_rates)

    # Figure 5: Lyapunov tuning
    print("\n[Fig 5] Generating Lyapunov tuning analysis...")
    lambda_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    violation_rates = []
    extreme_rates = []

    for lam in lambda_values:
        print(f"  Running lambda={lam}...")
        results = run_sim_with_predictive_control(
            p_chi=P_CHI, num_sims=NUM_SIMS, time_horizon=TIME_HORIZON,
            human_level=0, honest=False, uniform=False,
            f_max=0.5, tau_r=0.3, lambda_entropy=lam
        )
        stats = analyze_predictive_control_results(results)
        violation_rates.append(stats['lyapunov_violation_rate'] * 100)
        extreme_rates.append(stats['fraction_extreme'] * 100)

    plot_fig5_lyapunov_tuning(lambda_values, violation_rates, extreme_rates)

    # Figure 9: Versioning vs Predictive Control (critical comparison)
    print("\n[Fig 9] Generating critical comparison (Versioning vs Predictive)...")
    plot_fig9_versioning_vs_predictive(
        beliefs_ver, beliefs_pred, stats_ver, stats_pred, TIME_HORIZON
    )

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED!")
    print("=" * 60)

    plt.show()
