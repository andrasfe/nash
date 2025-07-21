# Nash Equilibrium Validation Using AI Agent Digital Twins

## Experiment Overview

This project validates a game-theoretic Nash equilibrium solution through **simulation using AI agents as digital twins**. We model participants in a federated insight system where individual AI agents make autonomous economic decisions, mimicking real human behavior under different incentive structures.

**The Experiment**: 100 AI agents receive personalized prompts describing their economic situation, game rules, and strategic options. Each agent independently decides whether to act honestly or maliciously based on their perceived incentives. The simulation tests whether our theoretical Nash equilibrium holds when faced with autonomous decision-making agents.

**Key Findings**: The AI agent simulation confirmed equilibrium stability under fair rotation with 3 temporary validators per round, validating the mathematical model with behavioral evidence.

## Player Definitions & Game Flow

### Player Types (Dynamic Roles)
All 100 participants are **identical at start** but receive **temporary role assignments** each round:

**🔄 Insight Generators (IGs)** - 97 per round
- **Role**: Generate local insights, vote on submissions
- **Economics**: Earn $0.00008/round, pay $0.0004 honest cost, stake $4.84
- **Behavior**: Choose honest (quality insights) vs malicious (low effort/gaming)
- **AI Agent Decisions**: "Do I invest effort in quality insights or try to game the system?"

**⚖️ Harmonizers (H)** - 3 per round (randomly selected)
- **Role**: Aggregate IG submissions, create final proposals, do mandatory IG work
- **Economics**: Earn ~$0.0047/round, pay $0.0001 honest cost, stake $161.28
- **Constraints**: Must spend 25% time on IG work (get 40x multiplier)
- **AI Agent Decisions**: "Do I fulfill my IG work quota honestly or try to skimp?"

### Round Flow (Each of 100 Rounds)

**1. Role Assignment** 🎯
- System selects 3 participants as temporary H validators based on:
  - Reputation scores (higher = more likely)
  - Fairness rotation (avoid consecutive selections)
  - Byzantine status (malicious agents penalized)
- Remaining 97 become IGs for this round

**2. IG Phase** 💡 
- Each IG receives personalized prompt with:
  - Current balance, reputation, role history
  - Game rules, costs, rewards, detection probabilities
  - Strategic temptations and consequences
- **AI Agent Decision**: Vote honestly (quality assessment) vs maliciously (random/biased)
- IGs submit votes on candidate submissions
- Pay voting costs, gain reputation if honest

**3. H Validator Phase** ⚖️
- Each of 3 H validators receives prompt with:
  - IG vote tallies and submission quality
  - Their mandatory IG work requirements (25% quota)
  - Economic incentives and penalties
- **AI Agent Decision**: Make honest aggregate decision vs game the system
- H validators must do IG work (get 40x multiplier) 
- Create collective final decision

**4. Economic Settlement** 💰
- Distribute rewards based on performance:
  - IGs: Base rate if honest, penalties if caught cheating
  - Hs: Higher rate + IG work bonuses if quota met
- Update balances, reputation scores
- Apply detection/penalties for malicious behavior (98% catch rate)

**5. Metrics Collection** 📊
- Track individual agent decisions and outcomes
- Monitor role distribution, earnings, reputation
- Validate Nash equilibrium conditions each round

### AI Agent Behavioral Models

**🎭 Persona Assignment** (using NVIDIA Nemotron-Personas dataset):
- Each of the 100 participants is assigned a unique persona from the Nemotron dataset
- Personas include rich demographic, professional, and personality information
- Examples: "disciplined and organized" retiree, "ambitious and competitive" professional, "creative and imaginative" artist
- Personality traits influence decision-making tendencies (e.g., risk-averse personas are more likely to choose safe strategies)

**🧠 Decision Factors** (each agent considers):
- Current economic position vs others
- Probability of detection if malicious
- Long-term reputation consequences 
- Fair rotation opportunities (when will I be H again?)
- Recent system events (Byzantine catches, punishments)

**📚 Memory & Learning**:
- Personal history of past roles and outcomes
- Observations of other agents' successes/failures
- System-wide trends (detection rates, earnings)
- Strategic adaptation over 100 rounds

This creates a **realistic behavioral laboratory** where economic theory meets autonomous agent decision-making under uncertainty.

## Problem Statement

In federated insight systems with two participant types:
- **Insight Generators (IG)**: Create local insights, earn $0.00008/round, stake $4.84
- **Harmonizers (H)**: Aggregate insights & create knowledge proposals, earn variable amounts, stake $161.28

Without proper incentive design, this creates a massive profit differential causing everyone to abandon IG work → system collapse.

## Solution: Fair Rotation + Two-Tier Incentive System

Our solution implements:

### 1. Fair Rotation System
- **3 temporary H validators** selected per round from all 100 participants
- Each participant serves as H validator ~3 times over 100 rounds
- Selection based on reputation + fairness (avoiding repeat selections)
- **Nash equilibrium**: π_H = 3/100 = 0.03 selection probability per round

### 2. Two-Tier Incentive Structure
- **IG Work Multiplier**: 40x rewards for harmonizers doing IG work (β_IG = 40)
- **Minimum Quota**: 25% of harmonizer time must be IG work (θ = 0.25)
- **Performance Bonuses**: $0.002 for meeting quotas (B_perf)
- **Adjusted Costs**: Reduced H honest cost to $0.0001 for fair rotation equilibrium

### Results
- **Profit ratio**: 58x (down from 312x) while maintaining validator incentives
- **Nash equilibrium**: ✅ All conditions satisfied (IGs prefer honest, Hs prefer honest)
- **Fair participation**: All 100 participants can serve as H validators over time
- **Behavioral validation**: AI agents consistently chose honest strategies

## Quick Start

```bash
# Step 1: Generate Nash equilibrium configuration
python nash_formulation.py 100 100  # Creates simulation_params_n100.json

# Step 2: Run simulation with AI agents (config file required)
python nash_simulation.py --config simulation_params_n100.json

# Optional: Run without LLM for faster testing
python nash_simulation.py --config simulation_params_n100.json --no-llm

# Optional: Specify custom output file
python nash_simulation.py --config simulation_params_n100.json --output my_results.json

# Test LLM connection
python test_llm_connection.py
```

## Core Files

| File | Description |
|------|-------------|
| `formulation.tex` | Mathematical Nash equilibrium proof with experimental validation |
| `nash_formulation.py` | Nash parameter calculation for fair rotation system |
| `nash_simulation.py` | AI agent simulation with temporary role assignments |
| `game_rules_prompt.py` | LLM prompts for agent decision-making |
| `persona_manager.py` | Loads and manages personas from NVIDIA Nemotron dataset |
| `simulation_params_n100.json` | Validated configuration for 100 participants, 100 rounds |
| `nash_results_*.json` | Simulation results with agent behavioral data |

## AI Agent Architecture

### Agent Decision Process
Each AI agent receives:
1. **Personal State**: Balance, reputation, stake locked, role history
2. **Game Rules**: Costs, rewards, detection probabilities, consequences  
3. **Current Context**: Round number, recent events, other participants' actions
4. **Strategic Options**: Honest vs malicious behavior with expected outcomes

### Behavioral Modeling
- **Persona-Based Variation**: Each agent's decisions are influenced by their assigned Nemotron persona
  - Competitive/ambitious personas: ~60% chance of risky behavior
  - Creative/young personas: ~50% chance of risky behavior
  - Disciplined/cautious personas: ~15% chance of risky behavior
- **Memory**: Each agent maintains history of past actions and outcomes
- **Temptation Modeling**: Agents consider short-term gains vs long-term reputation
- **Byzantine Simulation**: ~15% of agents are configured as Byzantine (malicious)

### Validation Metrics
- **Decision Consistency**: Do agents choose Nash equilibrium strategies?
- **Response to Incentives**: Do behavior changes match theoretical predictions?
- **Fair Rotation**: Do all participants get equal opportunities over time?
- **Equilibrium Stability**: Does the system remain stable over 100 rounds?

## Mathematical Foundation

The fair rotation Nash equilibrium ensures:

```
π_H = 3/100 = 0.03 (selection probability per participant per round)
E[U_IG(honest)] = $0.00008 > E[U_IG(malicious)] ✓
E[U_H(honest)] = $0.0047 > E[U_H(malicious)] = $0.0036 ✓
Profit ratio = 58x (manageable vs 312x problematic)
```

**Key insight**: With fair rotation, each participant gets selected as H validator ~3 times over 100 rounds, creating realistic promotion opportunities while maintaining system balance.

## Experimental Results

### Nash Equilibrium Validation ✅
- **IG honest preference**: $0.00008 vs -$0.0002 (honest wins)
- **H honest preference**: $0.0047 vs $0.0036 (honest wins)  
- **All equilibrium conditions**: Satisfied
- **Profit ratio**: Reduced from 312x to 58x

### AI Agent Behavioral Evidence
- **Honest strategy adoption**: >90% of non-Byzantine agents chose honest strategies
- **Byzantine detection**: ~98% of malicious behavior caught and penalized
- **Fair participation**: All 100 participants served as H validators over 100 rounds
- **Stable earnings**: Consistent per-round profits matching theoretical predictions

### System Stability
- **Role distribution**: Maintained 3 H / 97 IG split each round
- **Continuous operation**: No participant abandonment or system collapse
- **Economic sustainability**: Positive utility for honest behavior in both roles

## Setup

1. Clone the repository:
```bash
git clone https://github.com/andrasfe/nash.git
cd nash
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The simulation uses the NVIDIA Nemotron-Personas dataset which will be automatically downloaded on first run (~142MB).

3. Configure AI/LLM credentials:
```bash
cp .env.example .env
```

4. Edit `.env` with your API credentials:
```
API_KEY=your_api_key_here
BASE_URL=https://api.openai.com/v1/chat/completions
MODEL=gpt-4o-mini
```

5. Generate simulation parameters:
```bash
python nash_formulation.py 100 100  # 100 participants, 100 rounds
```

6. Run the simulation:
```bash
python nash_simulation.py --config simulation_params_n100.json
```

## Key Parameters (Fair Rotation System)

- **n_total = 100**: Total participants
- **n_h_per_round = 3**: Temporary H validators selected each round
- **num_rounds = 100**: Simulation length  
- **π_H = 0.03**: H selection probability per participant per round
- **β_IG = 40**: IG work reward multiplier
- **θ = 0.25**: Minimum IG work quota for H validators
- **Stakes**: IG: $4.84, H: $161.28
- **Costs**: IG honest: $0.0004, H honest: $0.0001 (adjusted for fair rotation)

## Recent Updates

### Simplified Workflow (2025-01)
- **Config-based execution**: Simulation now requires pre-generated config file from `nash_formulation.py`
- **Removed direct parameter calculation**: All Nash equilibrium parameters must be generated first
- **Cleaner separation**: Parameter generation (nash_formulation.py) vs simulation (nash_simulation.py)

### Fair Rotation Implementation (2025-01)
- **Removed permanent roles**: All participants are eligible for H validator selection
- **Temporary role assignment**: 3 H validators selected per round based on reputation + fairness
- **Selection algorithm**: Avoids repeat selections, ensures fair distribution over time
- **Nash recalibration**: Adjusted costs and rewards for π_H = 0.03 equilibrium

### AI Agent Validation
- **LLM integration**: Each participant is an autonomous AI agent making economic decisions
- **Persona diversity**: 100 unique personas from NVIDIA Nemotron dataset provide realistic behavioral variation
- **Behavioral testing**: Validated that agents choose Nash equilibrium strategies based on their personas
- **Digital twin confirmation**: Simulation results match theoretical predictions
- **Experimental evidence**: Documented in formulation.tex

### Parameter Optimization
- **H honest cost**: Reduced from $0.00208 to $0.0001 for fair rotation
- **Profit ratio**: Optimized to 58x (sustainable) vs 312x (problematic)
- **Selection probability**: π_H = 0.03 ensures realistic participation rates

## Citation

If you use this work in your research, please cite:

```bibtex
@software{ferenczi2025nash,
  title={Nash Equilibrium Validation Using AI Agent Digital Twins},
  author={Ferenczi, Andras \orcidID{0000-0001-6785-9416}},
  year={2025},
  url={https://github.com/andrasfe/nash},
  note={Game-theoretic mechanism design validated through simulation using AI agents as digital twins}
}
```

## License

This implementation is for research and educational purposes, demonstrating game-theoretic mechanism design validated through AI agent simulation.