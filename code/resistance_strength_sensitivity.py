"""
Sensitivity analysis for resistance_strength parameter.

Tests whether the 48x spiral differential and LPC criterion hold across
resistance_strength ρ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}.

This defends the Eq 30-31 behavioral assumptions by showing robustness
beyond the specific parameter choice.
"""

import sys
import io
import json
import numpy as np
from datetime import datetime

print("="*80)
print("RESISTANCE STRENGTH SENSITIVITY ANALYSIS")
print("="*80)
print()

# Suppress loading prints
print("Loading delusion2.py...")
old_stdout = sys.stdout
sys.stdout = io.StringIO()

from delusion2 import (
    run_sim_versioning_heterogeneous,
    analyze_versioning_heterogeneous_results,
    extract_final_beliefs_batch
)

sys.stdout = old_stdout
print("✓ Loaded\n")

# Configuration
NUM_SIMS = 1000
TIME_HORIZON = 50
P_CHI = 90
P_VALIDATION = 0.5
TYPE_CONFIDENCE_THRESHOLD = 0.7
WORK_THRESHOLD = 0.5

# Test range for resistance_strength
RESISTANCE_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print(f"Configuration:")
print(f"  N={NUM_SIMS}, T={TIME_HORIZON}, pχ={P_CHI}")
print(f"  p_validation={P_VALIDATION} (50% θ_V, 50% θ_G)")
print(f"  Testing resistance_strength ρ ∈ {RESISTANCE_VALUES}")
print()

# =============================================================================
# RUN SENSITIVITY ANALYSIS
# =============================================================================

results = []

for i, resistance in enumerate(RESISTANCE_VALUES, 1):
    print("="*80)
    print(f"[{i}/{len(RESISTANCE_VALUES)}] Testing resistance_strength = {resistance}")
    print("="*80)
    print()

    # Run simulation (one-turn protocol for baseline differential)
    sim_results = run_sim_versioning_heterogeneous(
        p_chi=P_CHI,
        num_sims=NUM_SIMS,
        time_horizon=TIME_HORIZON,
        human_level=0,
        honest=False,
        uniform=False,
        enable_auditor=True,
        enable_versioning=True,
        use_two_turn_protocol=False,
        p_validation=P_VALIDATION,
        resistance_strength=resistance,
        type_confidence_threshold=TYPE_CONFIDENCE_THRESHOLD
    )

    # Analyze with work-based classification
    stats = analyze_versioning_heterogeneous_results(
        sim_results,
        type_confidence_threshold=TYPE_CONFIDENCE_THRESHOLD,
        work_threshold=WORK_THRESHOLD
    )

    # Extract final beliefs for LPC check
    priors, _, _, _, _, _ = sim_results
    final_beliefs = extract_final_beliefs_batch(priors)
    mean_final_belief = float(final_beliefs.mean())
    overall_spiral_rate = float((final_beliefs > 0.9).mean())

    # Compute differential
    spiral_rate_g = stats['spiral_rate_growth']
    spiral_rate_v = stats['spiral_rate_validation']

    if spiral_rate_g > 0:
        differential = spiral_rate_v / spiral_rate_g
    else:
        differential = float('inf') if spiral_rate_v > 0 else 0.0

    # LPC check: mean_final_belief should be outside (0.45, 0.55)
    lpc_pass = mean_final_belief < 0.45 or mean_final_belief > 0.55

    # Store results
    result = {
        "resistance_strength": resistance,
        "n_growth": stats['n_growth'],
        "n_validation": stats['n_validation'],
        "spiral_rate_growth": spiral_rate_g,
        "spiral_rate_validation": spiral_rate_v,
        "differential": differential,
        "mean_final_belief": mean_final_belief,
        "overall_spiral_rate": overall_spiral_rate,
        "lpc_pass": lpc_pass,
        "work_separation": stats['work_separation'],
        "recall_growth": stats['recall_growth'],
        "recall_validation": stats['recall_validation'],
        "overall_detection_accuracy": stats['overall_detection_accuracy']
    }
    results.append(result)

    # Print summary
    print(f"Results for ρ = {resistance}:")
    print(f"  Population: {stats['n_growth']} θ_G, {stats['n_validation']} θ_V")
    print(f"  Spiral rate (θ_G): {spiral_rate_g:.1%}")
    print(f"  Spiral rate (θ_V): {spiral_rate_v:.1%}")
    print(f"  Differential:      {differential:.1f}x")
    print(f"  Mean final belief: {mean_final_belief:.3f}")
    print(f"  LPC status:        {'✅ PASS' if lpc_pass else '❌ FAIL'} (outside 0.45-0.55)")
    print(f"  Work separation:   {stats['work_separation']:.3f}")
    print()

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("="*80)
print("SUMMARY: Robustness Across Resistance Strength")
print("="*80)
print()

print("ρ     | Spiral θ_G | Spiral θ_V | Differential | Mean P̄   | LPC  | Work Sep")
print("------|------------|------------|--------------|----------|------|----------")
for r in results:
    rho = r['resistance_strength']
    sg = r['spiral_rate_growth']
    sv = r['spiral_rate_validation']
    diff = r['differential']
    mean_p = r['mean_final_belief']
    lpc = "PASS" if r['lpc_pass'] else "FAIL"
    wsep = r['work_separation']

    diff_str = f"{diff:.1f}x" if diff != float('inf') else "∞"

    print(f"{rho:.1f}   | {sg:>9.1%} | {sv:>9.1%} | {diff_str:>12} | {mean_p:>8.3f} | {lpc:>4} | {wsep:>8.3f}")

print()

# =============================================================================
# ROBUSTNESS ANALYSIS
# =============================================================================

print("="*80)
print("ROBUSTNESS ANALYSIS")
print("="*80)
print()

# Count how many values pass key criteria
lpc_pass_count = sum(1 for r in results if r['lpc_pass'])
high_differential_count = sum(1 for r in results if r['differential'] > 10)
positive_separation_count = sum(1 for r in results if r['work_separation'] > 0.1)

print(f"LPC Pass Rate:        {lpc_pass_count}/{len(results)} ({lpc_pass_count/len(results):.0%})")
print(f"High Differential:    {high_differential_count}/{len(results)} (>10x)")
print(f"Strong Separation:    {positive_separation_count}/{len(results)} (work_sep > 0.1)")
print()

# Check if results are robust
if lpc_pass_count >= 5 and high_differential_count >= 5:
    print("✅ ROBUST RESULT")
    print()
    print("The qualitative findings hold across most resistance values:")
    print("  • LPC criterion passes for majority of ρ values")
    print("  • Spiral differential remains large (>10x) across range")
    print("  • Behavioral assumptions in Eq 30-31 are not fragile")
    print()
    print("This demonstrates that separation exists for a wide range of")
    print("cost asymmetries, not just the specific illustrative values.")
elif lpc_pass_count >= 3 or high_differential_count >= 3:
    print("⚠️  MODERATE ROBUSTNESS")
    print()
    print("The results show some sensitivity to resistance_strength:")
    print(f"  • LPC passes for {lpc_pass_count}/{len(results)} values")
    print(f"  • High differential for {high_differential_count}/{len(results)} values")
    print()
    print("The mechanism works but may require careful parameter selection.")
else:
    print("❌ FRAGILE RESULT")
    print()
    print("The results are highly sensitive to resistance_strength:")
    print(f"  • LPC passes for only {lpc_pass_count}/{len(results)} values")
    print(f"  • High differential for only {high_differential_count}/{len(results)} values")
    print()
    print("This suggests the behavioral assumptions may be too specific.")

print()

# =============================================================================
# SAVE RESULTS
# =============================================================================

output = {
    "timestamp": datetime.now().isoformat(),
    "config": {
        "num_sims": NUM_SIMS,
        "time_horizon": TIME_HORIZON,
        "p_chi": P_CHI,
        "p_validation": P_VALIDATION,
        "resistance_values": RESISTANCE_VALUES,
        "type_confidence_threshold": TYPE_CONFIDENCE_THRESHOLD,
        "work_threshold": WORK_THRESHOLD
    },
    "results": results,
    "summary": {
        "lpc_pass_count": lpc_pass_count,
        "high_differential_count": high_differential_count,
        "positive_separation_count": positive_separation_count,
        "lpc_pass_rate": lpc_pass_count / len(results),
        "high_differential_rate": high_differential_count / len(results),
        "positive_separation_rate": positive_separation_count / len(results)
    }
}

output_file = "resistance_strength_sensitivity.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Results saved to {output_file}")
print()

# =============================================================================
# REVIEWER RESPONSE
# =============================================================================

print("="*80)
print("SUGGESTED REVIEWER RESPONSE")
print("="*80)
print()

if lpc_pass_count >= 5 and high_differential_count >= 5:
    print("\"Reviewer correctly notes cost parameters are illustrative. However,")
    print("sensitivity analysis (Appendix X) shows qualitative results are robust:")
    print()
    print(f"  • LPC criterion passes for {lpc_pass_count}/7 resistance values tested")
    print(f"  • Spiral differential >10x for {high_differential_count}/7 values")
    print(f"  • Work separation >0.1 for {positive_separation_count}/7 values")
    print()
    print("The formal contribution is the EXISTENCE of separating equilibria")
    print("under cost asymmetry (Proposition 2), not specific parameter values.\"")
else:
    print("The sensitivity analysis reveals moderate parameter dependence.")
    print("You should:")
    print("  1. Report full sensitivity results transparently")
    print("  2. Focus claims on the parameter ranges where mechanism works")
    print("  3. Acknowledge this as a limitation in Discussion section")

print()
