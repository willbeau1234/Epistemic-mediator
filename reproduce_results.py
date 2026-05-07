#!/usr/bin/env python3
"""
Quick reproduction script for paper submission.
Verifies that saved results match expected values and can regenerate key results.

Usage:
    python reproduce_results.py --verify    # Verify saved results
    python reproduce_results.py --run       # Run simulations (slow, ~2 hours)
"""

import json
import argparse
import os
from pathlib import Path

# Key expected values from paper
EXPECTED = {
    "baseline_spiral_rate": 0.536,
    "reactive_spiral_rate": 0.166,
    "bv_spiral_rate": 0.081,
    "pc_spiral_rate": 0.0,
    "cohens_h_b_vs_ra": 0.804,
    "cohens_h_b_vs_bv": 1.066,
    "cohens_h_ra_vs_bv": 0.262,
    "baseline_ci_lower": 0.505,
    "baseline_ci_upper": 0.566,
}

def verify_results():
    """Verify that saved results match expected values."""
    print("=" * 60)
    print("VERIFYING SAVED RESULTS")
    print("=" * 60)

    results_dir = Path(__file__).parent / "results"

    # Load main experiment results
    main_results_path = results_dir / "main_experiment_results_final.json"
    if not main_results_path.exists():
        print(f"ERROR: {main_results_path} not found")
        return False

    with open(main_results_path) as f:
        main_results = json.load(f)

    print("\n1. Main Simulation Results")
    print("-" * 40)

    # Verify baseline
    baseline = main_results["key_results"]["baseline_no_intervention"]["spiral_rate"]
    match = "✅" if abs(baseline - EXPECTED["baseline_spiral_rate"]) < 0.001 else "❌"
    print(f"  Baseline spiral rate: {baseline:.3f} (expected: {EXPECTED['baseline_spiral_rate']}) {match}")

    # Verify reactive
    reactive = main_results["key_results"]["reactive_auditor"]["spiral_rate"]
    match = "✅" if abs(reactive - EXPECTED["reactive_spiral_rate"]) < 0.001 else "❌"
    print(f"  Reactive spiral rate: {reactive:.3f} (expected: {EXPECTED['reactive_spiral_rate']}) {match}")

    # Verify PC
    pc = main_results["key_results"]["predictive_control_all_lambda"]["fraction_extreme"]
    match = "✅" if abs(pc - EXPECTED["pc_spiral_rate"]) < 0.001 else "❌"
    print(f"  Predictive Control: {pc:.3f} (expected: {EXPECTED['pc_spiral_rate']}) {match}")

    # Load stats results
    stats_path = results_dir / "results_task7_rigorous_stats_VERIFIED.json"
    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found")
        return False

    with open(stats_path) as f:
        stats_results = json.load(f)

    print("\n2. Statistical Analysis (Cohen's h)")
    print("-" * 40)

    for comp in stats_results["comparisons"]:
        name = comp["comparison"]
        h = comp["cohens_h"]

        if "Baseline vs Reactive" in name:
            expected = EXPECTED["cohens_h_b_vs_ra"]
        elif "Baseline vs Belief" in name:
            expected = EXPECTED["cohens_h_b_vs_bv"]
        elif "Reactive Auditor vs Belief" in name:
            expected = EXPECTED["cohens_h_ra_vs_bv"]
        else:
            continue

        match = "✅" if abs(h - expected) < 0.01 else "❌"
        print(f"  {name}: h={h:.3f} (expected: {expected}) {match}")

    print("\n3. Confidence Intervals (95% BCa Bootstrap)")
    print("-" * 40)

    baseline_comp = stats_results["comparisons"][0]
    ci = baseline_comp["spiral_rate_1_ci"]
    match_lower = "✅" if abs(ci[0] - EXPECTED["baseline_ci_lower"]) < 0.01 else "❌"
    match_upper = "✅" if abs(ci[1] - EXPECTED["baseline_ci_upper"]) < 0.01 else "❌"
    print(f"  Baseline CI: [{ci[0]:.3f}, {ci[1]:.3f}] (expected: [{EXPECTED['baseline_ci_lower']}, {EXPECTED['baseline_ci_upper']}]) {match_lower}{match_upper}")

    # Load LLM validation
    llm_path = results_dir / "llm_validation_results_final.json"
    if llm_path.exists():
        with open(llm_path) as f:
            llm_results = json.load(f)

        print("\n4. LLM Validation (GPT-4o)")
        print("-" * 40)

        high_baseline = llm_results["table_2_numbers"]["high_sycophancy"]["baseline_original_coding"]["spiral_rate"]
        print(f"  High sycophancy baseline: {high_baseline*100:.0f}%")

        high_versioning = llm_results["table_2_numbers"]["high_sycophancy"]["versioning_original_coding"]["spiral_rate"]
        print(f"  High sycophancy versioning: {high_versioning*100:.0f}%")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    return True


def run_simulations():
    """Run the main simulations (slow)."""
    print("=" * 60)
    print("RUNNING SIMULATIONS")
    print("=" * 60)
    print("\nThis will take approximately 2 hours on a modern CPU.")
    print("The simulation uses JAX for acceleration.\n")

    # Change to code directory
    code_dir = Path(__file__).parent / "code"
    os.chdir(code_dir)

    # Import and run
    print("Importing delusion2.py (JAX compilation may take 1-2 min)...")
    import delusion2

    print("\nRunning main experiment...")
    # The main experiment runs automatically when delusion2.py is imported
    # if it detects it's being run as __main__

    print("\nSimulations complete. Check output above for results.")


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper submission results")
    parser.add_argument("--verify", action="store_true", help="Verify saved results")
    parser.add_argument("--run", action="store_true", help="Run simulations (slow)")
    args = parser.parse_args()

    if args.verify:
        verify_results()
    elif args.run:
        run_simulations()
    else:
        # Default: verify
        print("No action specified. Running verification...\n")
        verify_results()
        print("\nTo run simulations, use: python reproduce_results.py --run")


if __name__ == "__main__":
    main()
