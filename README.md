# Epistemic Mediator with Belief Versioning

## Paper Title
**Playing games with knowledge: AI-Induced delusions need game theoretic interventions**

---

## Quick Start: Reproducing Results

### Prerequisites
```bash
# Python 3.10+ required
pip install -r requirements.txt

# Or install manually:
pip install jax jaxlib numpy scipy matplotlib tqdm scikit-learn
pip install memo-lang  # Probabilistic programming framework (Chandra et al., 2025)

# For LLM experiments (optional, ~$20 API cost):
pip install openai
```

**Note:** The `memo` library is a probabilistic programming framework. See [memo documentation](https://github.com/probcomp/memo) for details.

### Main Results (Table 1)
```bash
cd code/
python3 delusion2.py
```
**Expected Output:** Baseline 53.6%, Reactive 16.6%, BV 8.1%, PC 0.0%

**Precomputed Results:** `results/main_experiment_results_final.json`

---

## Complete Number-to-Source Mapping

### ABSTRACT CLAIMS

| Paper Claim | Value | Source File | Verification |
|-------------|-------|-------------|--------------|
| Baseline spiral rate | **53.6%** | `results/main_experiment_results_final.json` line 15 | `"spiral_rate": 0.536` |
| BV spiral rate | **8.1%** | `results/main_experiment_results_final.json` line 85 | `"belief_versioning_spiral_rate": "8.1%"` |
| Reduction | **85%** | Computed: `(53.6-8.1)/53.6 = 0.849` | Verified |
| Mean belief (BV) | **P̄=0.32** | `results/main_experiment_results_final.json` | BV passes LPC |
| Cohen's h (vs baseline) | **1.066** | `results/results_task7_rigorous_stats_VERIFIED.json` line 52 | `"cohens_h": 1.0656...` |
| 20 heuristic agents | **34.4pp reduction** | `code/task2b_directional_agents.py` output | Extended agent library |
| Min cost ratio | **1.5×** | `results/resistance_strength_sensitivity.json` | Sweep results |
| GPT-4o BV | **16.5%** | `results/llm_validation_results_final.json` | Cross-paradigm validation |
| GPT-4o RA | **47%** | `results/llm_validation_results_final.json` | Cross-paradigm validation |

---

### TABLE 1: Simulation Intervention Methods

| Method | Spiral Rate | Source | Line in JSON |
|--------|-------------|--------|--------------|
| No Auditor (Baseline) | **53.6%** | `main_experiment_results_final.json` | `baseline_no_intervention.spiral_rate: 0.536` |
| Reactive Auditor | **16.6%** | `main_experiment_results_final.json` | `reactive_auditor.spiral_rate: 0.166` |
| Belief Versioning | **8.1%** | `main_experiment_results_final.json` | `paper_table_1_numbers.belief_versioning_spiral_rate` |
| Predictive Control | **0.0%** | `main_experiment_results_final.json` | `predictive_control_all_lambda.fraction_extreme: 0.000` |

**Cohen's h Effect Sizes** (from `results_task7_rigorous_stats_VERIFIED.json`):
- Baseline vs Reactive: **h = 0.804** (large effect)
- Baseline vs BV: **h = 1.066** (large effect)
- RA vs BV: **h = 0.262** (small effect)

**95% Confidence Intervals** (BCa bootstrap, n=10,000):
- Baseline: **[50.5%, 56.6%]**
- Reactive Auditor: **[14.3%, 18.9%]**
- Belief Versioning: **[6.4%, 9.8%]**

---

### TABLE 2: LLM Validation (GPT-4o)

| Setting | Intervention | Spiral Rate | P̄ | Source |
|---------|--------------|-------------|---|--------|
| Simulation | No Auditor | **53.6%** | 0.54 | `main_experiment_results_final.json` |
| Simulation | Reactive | **16.6%** | 0.40 | `main_experiment_results_final.json` |
| Simulation | BV | **8.1%** | 0.32 | `main_experiment_results_final.json` |
| GPT-4o | No Auditor | **100%** | 1.000 | `llm_validation_results_final.json` |
| GPT-4o | Reactive | **47%** | 0.875 | `llm_validation_results_final.json` |
| GPT-4o | BV | **16.5%** | 0.821 | `llm_validation_results_final.json` |

---

### SECTION 5.2: Heuristic Agent Generalization

| Claim | Value | Source |
|-------|-------|--------|
| 4 canonical agent types | **47.4pp reduction** | `heuristic_agent_results.json` |
| 20 total agent types | **34.4pp reduction** | `task2b_directional_agents.py` |
| Spiral rate ratio (ρ sweep) | **1.8×–4.2×** | `resistance_strength_sensitivity.json` |

---

### SECTION 5.4: Robustness Analyses

**Cost Asymmetry** (from `resistance_strength_sensitivity.json`):
- Minimum effective ratio: **1.5×**
- Literature-grounded ratio (d≈0.70): **~1.86×**
- 8/9 configurations pass separating equilibrium + LPC

**Extreme Sycophancy** (p_χ ∈ {0.91, 0.93, 0.95, 0.97, 0.99}):
- Average reduction: **~35pp**
- LPC pass rate: **100%**

---

## Simulation Parameters

All simulations use these default parameters (from `code/delusion2.py`):

```python
# Global configuration
num_sims = 1000           # Number of Monte Carlo simulations
time_horizon = 50         # Conversation turns
seed = 42                 # Random seed for reproducibility
p_chi = 0.9               # 90% sycophancy level

# Detection thresholds
tau_v = 0.02              # Entrenchment velocity threshold
tau_h = -0.05             # Entropy decay threshold

# Friction parameters
friction = 0.3            # Default friction level (F*)
friction_beta = 1.5       # Cost scaling for validation-seekers

# Belief Versioning parameters
entropy_min = 1.0         # Minimum entropy for commit
velocity_epsilon = 0.02   # Maximum velocity for commit
belief_delta = 0.3        # Belief boundary (commits only in [0.3, 0.7])
type_confidence_threshold = 0.7  # γ* for type classification

# Predictive Control parameters
f_max = 0.5               # Maximum friction
tau_r = 0.3               # Risk threshold
lambda_entropy = 0.1      # Lyapunov regularization weight
```

**LLM Validation Configuration:**
```python
model = "gpt-4o"
temperature = 0.7
n = 200                   # Simulations (20 per seed × 10 seeds, or 3 seeds)
time_horizon = 30
seeds = [5000, 5500, 6000]
```

---

## File Structure

```
epistemic_mediator/
├── README.md                 # This file
├── code/
│   ├── delusion2.py          # Main simulation engine (JAX)
│   ├── delusion.py           # Original simulation (backup)
│   ├── make_plot2.py         # Figure generation
│   ├── make_plot.py          # Alternative plotter
│   ├── analyze_n200_results.py        # Statistical analysis
│   ├── task2_heuristic_agents.py      # 4 canonical heuristic agents
│   ├── task2b_directional_agents.py   # 20 extended agent types
│   ├── resistance_strength_sensitivity.py  # Cost ratio sweep
│   ├── run_gpt4o_experiment.py        # LLM baseline
│   ├── run_gpt4o_belief_versioning.py # LLM + BV
│   ├── run_gpt4o_reactive_auditor.py  # LLM + RA
│   ├── gpt4o_bot_wrapper.py           # GPT-4o API wrapper
│   └── Delusional2_LLM.py             # LLM experiment utilities
│
├── results/
│   ├── main_experiment_results_final.json    # Primary simulation results
│   ├── results_task7_rigorous_stats_VERIFIED.json  # Statistical analysis
│   ├── llm_validation_results_final.json     # GPT-4o validation results
│   ├── resistance_strength_sensitivity.json  # Cost ratio experiments
│   ├── heuristic_agent_results.json          # Heuristic agent experiments
│   ├── conservative_coding_results.json      # Coding sensitivity analysis
│   ├── two_turn_heterogeneous_results_corrected.json
│   └── effect_size_audit.json
│
└── figures/
    ├── fig1_8_combined.{pdf,png}        # Figure 1: Reactive Auditor
    ├── fig2_belief_versioning.{pdf,png} # Figure 2: Belief Versioning
    ├── fig3_predictive_control.{pdf,png}# Figure 3: Predictive Control
    ├── fig4_method_comparison.{pdf,png} # Figure 4: Method comparison
    ├── fig9_versioning_vs_predictive.{pdf,png}  # Learning Preservation
    ├── fig_llm_comparison.{pdf,png}     # LLM validation comparison
    ├── fig_llm_vs_simulation.{pdf,png}  # Cross-paradigm consistency
    └── fig_task7_rigorous_stats_VERIFIED.{pdf,png}  # Statistical summary
```

---

## Reproducing Specific Results

### 1. Main Simulation Results (Table 1)
```bash
cd code/
python3 delusion2.py
# Output includes: Baseline 53.6%, Reactive 16.6%, BV 8.1%, PC 0.0%
# Runtime: ~2 hours on M-series Mac
```

### 2. Statistical Analysis (Cohen's h, CIs)
```bash
cd code/
python3 analyze_n200_results.py
# Generates: Cohen's h values, BCa bootstrap CIs, Bonferroni-corrected p-values
```

### 3. Heuristic Agent Experiments
```bash
cd code/
python3 task2_heuristic_agents.py      # 4 canonical types
python3 task2b_directional_agents.py   # 20 extended types
# Output: Spiral rate reductions per agent type
```

### 4. Cost Ratio Sensitivity
```bash
cd code/
python3 resistance_strength_sensitivity.py
# Tests ρ ∈ {0.3, 0.4, ..., 0.9}
# Verifies 1.5× minimum ratio
```

### 5. LLM Validation (requires API key, ~$20)
```bash
export OPENAI_API_KEY="your-key"
cd code/
python3 run_gpt4o_experiment.py            # Baseline
python3 run_gpt4o_belief_versioning.py     # BV condition
python3 run_gpt4o_reactive_auditor.py      # RA condition
```

### 6. Figure Generation
```bash
cd code/
python3 make_plot2.py
# Generates all figures using saved results
```

---

## Key Code Functions

### `delusion2.py` - Main Simulation Engine

| Function | Description | Paper Section |
|----------|-------------|---------------|
| `run_sim_with_auditor()` | Reactive Auditor simulation | Section 5.1 |
| `run_sim_with_belief_versioning()` | Belief Versioning (main contribution) | Section 5.1 |
| `run_sim_with_predictive_control()` | Predictive Control baseline | Section 5.1 |
| `run_sim_heterogeneous_types()` | Heterogeneous agent simulation | Section 5.2 |
| `fit_alpha_parameters()` | Risk classifier training | Section 4.4 |
| `extract_final_beliefs_batch()` | Extract P̄ from simulations | Tables 1-2 |

### Detection Mechanisms

```python
# Entrenchment velocity (V_e)
v_e = np.mean(np.diff(belief_hist[-4:]))

# Entropy decay (ΔH)
delta_h = np.mean(np.diff(entropy_hist[-4:]))

# Trigger condition: T = 𝕀[V_e > τ_v ∧ ΔH < τ_h]
trigger = (v_e > tau_v) & (delta_h < tau_h)

# Friction intervention
P_corrected = (1 - F) * P_bayes + F * P_uniform
```

---

## Verification Checklist

- [x] Baseline spiral rate: 53.6% (verified in JSON)
- [x] Reactive Auditor: 16.6% (verified in JSON)
- [x] Belief Versioning: 8.1% (verified in JSON)
- [x] Cohen's h values: 0.804, 1.066, 0.262 (verified in stats JSON)
- [x] 95% CIs match paper (verified in stats JSON)
- [x] LLM validation numbers (verified in llm JSON)
- [x] All parameters documented (see above)
- [x] Random seed: 42 (ensures reproducibility)

---

## Citation

If you use this code, please cite:
```bibtex
@article{anonymous2026epistemic,
  title={Playing games with knowledge: AI-Induced delusions need game theoretic interventions},
  author={Anonymous},
  year={2026}
}
```

---

## Contact

For questions about reproducing results, contact the authors.

**Package Generated:** May 6, 2026
**Verification Status:** All numbers verified against source code and data files
