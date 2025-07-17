# Nash Equilibrium Solver for Blockchain Validation Systems

This repository implements a Nash equilibrium solver for a blockchain validation incentive system with reputation-based validator emergence. The system models a two-player game between Insight Generators (IGs) and Validators (Hs) with economic incentives designed to ensure honest behavior.

## Overview

The system demonstrates how a blockchain network can achieve a sustainable equilibrium where:
- All participants start as Insight Generators (IGs)
- After 100 correct contributions (~1.8 hours), IGs can become validators
- Economic incentives ensure honest behavior through a Nash equilibrium
- The network maintains 100 validators from 1,000 participants with reasonable stakes
- Daily network cost is kept at a sustainable $1,843

## Symbol Definitions

### Network Parameters
- **n**: Number of Insight Generators (e.g., 1,000)
- **n_H**: Number of active validators (e.g., 100)
- **k**: Validators selected per round (e.g., 5)
- **r**: Reputation score (+1 per correct contribution)
- **p**: False-negative rate for malicious detection (0.02 = 2%)
- **π_H**: Selection probability for validators (k/n_H = 0.05)

### Economic Parameters
- **R_IG**: Reward per round for IGs ($0.00048)
- **R_H**: Reward per round for selected validators ($0.16)
- **C_IG^hon**: Cost for honest IG behavior ($0.0004)
- **C_IG^mal**: Cost for malicious IG behavior ($0.0002)
- **C_H^hon**: Cost for honest validator behavior ($0.04)
- **C_H^mal**: Cost for malicious validator behavior ($0.008)
- **S_IG**: Stake required from IGs ($4.84)
- **S_H**: Stake required from validators ($161.28)

### Utility Functions
- **U_IG(α)**: Utility for IG with strategy α ∈ {hon, mal}
  ```
  U_IG(α) = R_IG - C_IG^α - 𝟙_{α=mal}(1-p)(R_IG + S_IG)
  ```
- **U_H(α_H)**: Utility for validator with strategy α_H
  ```
  U_H(α_H) = U_IG(hon) + π_H·R_H - C_H^α_H - 𝟙_{α_H=mal}(1-p)(R_H + S_H)
  ```

## Nash Equilibrium

The system achieves Nash equilibrium when no player can improve their utility by unilaterally changing strategy.

### IG Nash Condition
For IGs to remain honest:
```
S_IG ≥ (C_IG^hon - C_IG^mal)/(1-p) - R_IG
S_IG ≥ (0.0004 - 0.0002)/0.98 - 0.00048 ≈ -0.000276
```

### Validator Nash Condition
For validators to remain honest:
```
S_H ≥ (C_H^hon - C_H^mal)/(1-p) - R_H
S_H ≥ (0.04 - 0.008)/0.98 - 0.16 ≈ -0.127
```

Note: The negative minimum stakes indicate that the economic incentives are so strong that participants would be honest even without stakes. The actual stakes provide additional security margin.

## Key Results

- **Daily Network Cost**: $1,843
- **IG:Validator Ratio**: 10:1 (100 validators from 1,000 participants)
- **Time to Validator Status**: 1.8 hours (100 contributions at 95% success rate)
- **IG Margin**: 20% profit on costs
- **Validator Margin**: 300% profit when selected
- **Security Value**: $20,990 (combined stakes + reputation value)

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: If you encounter numpy compatibility issues with cvxpy, run:
```bash
pip install "numpy<2"
```

## Running the Simulation

```bash
python reputation_validator_example.py
```

The simulation will:
1. Run for 30 simulated days (720 hours)
2. Track the emergence of validators from the IG pool
3. Verify Nash equilibrium conditions
4. Generate visualization plots
5. Output key metrics and insights

### Simulation Parameters

The simulation uses the following default parameters (configurable in the code):
- Initial IGs: 1,000
- Reputation threshold: 100 correct contributions
- Rounds per hour: 60 (1 per minute)
- Correct contribution rate: 95%
- Simulation duration: 30 days

## Output Files

- `reputation_network_evolution.png`: Shows network growth over time
- `reputation_network_analysis.png`: Displays key metrics and trends

## Mathematical Formulation

See `formulation.tex` and `formulation2.tex` for the complete mathematical proof of the Nash equilibrium, including:
- Utility function definitions
- Nash condition derivations
- System limitations and extensions
- Byzantine resilience properties (n ≥ 3f+1)

## How It Works

1. **Reputation Building**: IGs contribute insights and earn +1 reputation per correct contribution
2. **Validator Emergence**: After 100 correct contributions, IGs become eligible to validate
3. **Economic Decision**: Eligible IGs choose to validate based on profitability
4. **Nash Equilibrium**: Stakes and rewards ensure honest behavior is always optimal
5. **Network Stability**: The system reaches equilibrium with ~100 validators from 1,000 participants

## Key Insights

1. **Reputation as Security**: The 100-contribution threshold creates natural sybil resistance
2. **Sustainable Economics**: Daily cost of $1,843 supports 1,000 participants and 100 validators
3. **Strong Nash Equilibrium**: Stakes far exceed minimum requirements for security
4. **Organic Growth**: Validators emerge naturally based on merit, not just capital
5. **Byzantine Resilience**: System maintains n ≥ 3f+1 for fault tolerance

## Limitations

1. Single-deviator Nash only (coalitions could have higher effective p)
2. Fixed false-negative rate (should adapt to attack patterns)
3. No time discounting in utility functions
4. Capped stakes may reduce security for larger networks
5. Assumes purely rational actors

## Future Extensions

- Coalition-proof Nash equilibrium
- Adaptive detection rates
- Time-discounted utilities
- Dynamic stake adjustment
- Behavioral economics integration

## License

This implementation is for research and educational purposes.