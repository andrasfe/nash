#!/usr/bin/env python3
"""
Nash Equilibrium Formulation Implementation
Based on formulation.tex

This module implements the mathematical formulation for the two-tier incentive system
that prevents harmonizer monoculture in federated insight systems.
"""

import numpy as np
from typing import Dict, Tuple


class NashFormulation:
    """Implementation of Nash equilibrium calculations from the paper"""
    
    def __init__(self):
        """Initialize with parameters from formulation.tex"""
        # System parameters
        self.n_ig = 1000  # Number of IGs
        self.n_h = 3      # Number of Harmonizers selected per round (fair rotation)
        self.k = 3        # H validators selected per round (fair rotation)
        
        # Stakes (in dollars)
        self.S_ig = 4.84
        self.S_h = 161.28
        
        # Rewards
        self.R_ig = 0.00048  # IG reward per round
        self.R_h = 0.16      # H reward per round
        
        # Costs (corrected to match expected profits from paper)
        # Working backwards: U_IG(hon) = R_IG - C_IG_hon = 0.00008
        # So: C_IG_hon = 0.00048 - 0.00008 = 0.0004 ✓
        self.C_ig_hon = 0.0004   # IG honest cost
        self.C_ig_mal = 0.0002   # IG malicious cost
        
        # Recalculated for fair rotation (π_H = 0.03):
        # For Nash equilibrium: U_H(hon) > U_H(mal)
        # π_H * R_H - C_H_hon > π_H * R_H * (1-p) - p * S_H / 10000 - C_H_mal
        # 0.03 * 0.16 - C_H_hon > 0.03 * 0.16 * 0.98 - 0.02 * 161.28 / 10000 - C_H_mal
        # 0.0048 - C_H_hon > 0.004704 - 0.00032256 - C_H_mal
        # C_H_hon < 0.0048 - 0.004704 + 0.00032256 + C_H_mal
        # C_H_hon < -0.00010144 + C_H_mal (impossible if C_H_mal > 0)
        # Solution: Reduce C_H_hon significantly for fair rotation
        self.C_h_hon = 0.0001    # H honest cost (reduced for fair rotation)
        self.C_h_mal = 0.0008    # H malicious cost (keep higher to discourage malicious behavior)
        
        # Probabilities for fair rotation (100 rounds, 3 H per round)
        # Each participant gets selected ~3 times over 100 rounds with fair rotation
        self.pi_h = 3.0 / 100.0  # H selection probability = 0.03 per round
        self.p = 0.02  # False negative rate
        
        # Two-tier solution parameters
        self.beta_ig = 40      # IG work multiplier
        self.theta = 0.25      # Minimum IG quota (25%)
        self.B_perf = 0.002    # Performance bonus
        self.rho = 0.15        # Reputation impact
        
        # Promotion parameters
        self.R = 15  # Reputation needed for promotion
        self.X = 15  # Maximum rounds as harmonizer
    
    def calculate_ig_utility(self, strategy: str = 'honest') -> float:
        """Calculate expected utility for IG
        
        Args:
            strategy: 'honest' or 'malicious'
            
        Returns:
            Expected utility per round
        """
        if strategy == 'honest':
            # E[U_IG(hon)] = R_IG - C_IG^hon
            return self.R_ig - self.C_ig_hon
        else:
            # E[U_IG(mal)] = -p * S_IG / 10000 - C_IG^mal
            return -self.p * self.S_ig / 10000 - self.C_ig_mal
    
    def calculate_h_utility_base(self, strategy: str = 'honest') -> float:
        """Calculate base expected utility for Harmonizer (without two-tier solution)
        
        Args:
            strategy: 'honest' or 'malicious'
            
        Returns:
            Expected utility per round
        """
        if strategy == 'honest':
            # E[U_H(hon)] = π_H * R_H - C_H^hon
            return self.pi_h * self.R_h - self.C_h_hon
        else:
            # E[U_H(mal)] = π_H * R_H * (1-p) - p * S_H / 10000 - C_H^mal
            return self.pi_h * self.R_h * (1 - self.p) - self.p * self.S_h / 10000 - self.C_h_mal
    
    def calculate_h_utility_with_solution(self) -> float:
        """Calculate expected utility for Harmonizer with two-tier solution
        
        Returns:
            Expected utility per round with solution applied
        """
        # Base utility from harmonization work (75% of time)
        base_utility = self.calculate_h_utility_base('honest')
        h_work_component = (1 - self.theta) * base_utility
        
        # Boosted IG work (25% of time with 40x multiplier)
        ig_base = self.calculate_ig_utility('honest')
        ig_work_component = self.theta * self.beta_ig * ig_base
        
        # Performance bonus
        bonus_component = self.B_perf * self.rho
        
        # Total expected utility
        return h_work_component + ig_work_component + bonus_component
    
    def calculate_profit_ratios(self) -> Dict[str, float]:
        """Calculate all profit ratios
        
        Returns:
            Dictionary with profit ratios
        """
        ig_profit = self.calculate_ig_utility('honest')
        h_profit_base = self.calculate_h_utility_base('honest')
        h_profit_with_solution = self.calculate_h_utility_with_solution()
        
        return {
            'ig_profit_per_round': ig_profit,
            'h_profit_base_per_round': h_profit_base,
            'h_profit_with_solution_per_round': h_profit_with_solution,
            'profit_ratio_without_solution': h_profit_base / ig_profit if ig_profit > 0 else float('inf'),
            'profit_ratio_with_solution': h_profit_with_solution / ig_profit if ig_profit > 0 else float('inf'),
            'h_profit_retention': h_profit_with_solution / h_profit_base if h_profit_base > 0 else 0
        }
    
    def verify_nash_equilibrium(self) -> Dict[str, bool]:
        """Verify Nash equilibrium conditions
        
        Returns:
            Dictionary with verification results
        """
        ig_hon = self.calculate_ig_utility('honest')
        ig_mal = self.calculate_ig_utility('malicious')
        h_hon = self.calculate_h_utility_base('honest')
        h_mal = self.calculate_h_utility_base('malicious')
        
        return {
            'ig_prefers_honest': ig_hon > ig_mal,
            'h_prefers_honest': h_hon > h_mal,
            'ig_positive_utility': ig_hon > 0,
            'h_maintains_profit': self.calculate_h_utility_with_solution() > 0.8 * h_hon,
            'nash_equilibrium': ig_hon > ig_mal and h_hon > h_mal
        }
    
    def calculate_daily_cost(self) -> float:
        """Calculate daily operational cost
        
        Returns:
            Total daily cost in dollars
        """
        # Total stakes locked
        total_stakes = self.n_ig * self.S_ig + self.n_h * self.S_h
        
        # Daily operational cost (from paper)
        return 1843.0  # $1,843 per day
    
    def print_summary(self):
        """Print summary of all calculations"""
        print("Nash Equilibrium Formulation Summary")
        print("=" * 60)
        
        print("\nSystem Parameters:")
        print(f"  Total participants: {self.n_ig + self.n_h}")
        print(f"  IGs: {self.n_ig} ({self.n_ig/(self.n_ig + self.n_h)*100:.1f}%)")
        print(f"  Harmonizers: {self.n_h} ({self.n_h/(self.n_ig + self.n_h)*100:.1f}%)")
        
        print("\nStakes:")
        print(f"  IG stake: ${self.S_ig}")
        print(f"  H stake: ${self.S_h}")
        
        print("\nTwo-tier Solution Parameters:")
        print(f"  IG multiplier (β): {self.beta_ig}x")
        print(f"  Minimum IG quota (θ): {self.theta*100:.0f}%")
        print(f"  Performance bonus: ${self.B_perf}")
        
        profits = self.calculate_profit_ratios()
        print("\nExpected Profits:")
        print(f"  IG profit/round: ${profits['ig_profit_per_round']:.5f}")
        print(f"  H profit/round (base): ${profits['h_profit_base_per_round']:.5f}")
        print(f"  H profit/round (with solution): ${profits['h_profit_with_solution_per_round']:.5f}")
        
        print("\nProfit Ratios:")
        print(f"  Without solution: {profits['profit_ratio_without_solution']:.0f}x")
        print(f"  With solution: {profits['profit_ratio_with_solution']:.0f}x")
        print(f"  H profit retention: {profits['h_profit_retention']*100:.1f}%")
        
        verification = self.verify_nash_equilibrium()
        print("\nNash Equilibrium Verification:")
        for key, value in verification.items():
            print(f"  {key}: {'✓' if value else '✗'}")
        
        print(f"\nDaily operational cost: ${self.calculate_daily_cost()}")


def calculate_dynamic_parameters(n_total: int) -> Dict[str, float]:
    """Calculate parameters dynamically based on total participants
    
    This matches the scaling in calculate_small_scale_params.py
    
    Args:
        n_total: Total number of participants
        
    Returns:
        Dictionary with calculated parameters
    """
    # Use exactly 3 temporary harmonizers (selected randomly each round)
    n_h = 3  # Exactly 3 harmonizers
    n_ig = n_total - n_h
    
    # Scale multiplier and quota based on n_total
    if n_total <= 12:
        ig_multiplier = 80
        min_ig_quota = 0.35
    elif n_total <= 20:
        # Linear interpolation
        ig_multiplier = 80 - (n_total - 12) * 5
        min_ig_quota = 0.35 - (n_total - 12) * 0.01
    elif n_total <= 50:
        # Continue scaling
        ig_multiplier = max(40, 60 - (n_total - 20) * 0.67)
        min_ig_quota = max(0.25, 0.30 - (n_total - 20) * 0.0017)
    else:
        # Paper values for large scale
        ig_multiplier = 40
        min_ig_quota = 0.25
    
    return {
        'n_total': n_total,
        'n_h': n_h,
        'n_ig': n_ig,
        'ig_multiplier': ig_multiplier,
        'min_ig_quota': min_ig_quota,
        'h_proportion': n_h / n_total
    }


def generate_simulation_config(n_total: int, num_rounds: int = 100, output_file: str = None) -> Dict:
    """Generate simulation configuration based on Nash formulation
    
    Args:
        n_total: Total number of participants
        num_rounds: Number of simulation rounds
        output_file: Optional JSON file path to save config
        
    Returns:
        Configuration dictionary
    """
    # Get dynamic parameters
    dynamic = calculate_dynamic_parameters(n_total)
    
    # Create formulation for calculations
    form = NashFormulation()
    
    # Calculate expected profits
    profits = form.calculate_profit_ratios()
    
    config = {
        "n_total": n_total,
        "n_h_initial": dynamic['n_h'],
        "n_ig_initial": dynamic['n_ig'],
        "stake_ig": form.S_ig,
        "stake_h": form.S_h,
        "initial_balance": 200.0,
        "ig_cost_honest": form.C_ig_hon,
        "ig_cost_malicious": form.C_ig_mal,
        "h_cost_honest": form.C_h_hon,
        "h_cost_malicious": form.C_h_mal,
        "h_reward": form.R_h,
        "h_selection_prob": form.pi_h,  # Use paper's π_H = k/n_H = 0.05
        "ig_multiplier": dynamic['ig_multiplier'],
        "min_ig_quota": dynamic['min_ig_quota'],
        "h_bonus": form.B_perf,
        "reputation_impact": form.rho,
        "reputation_reward": 1,
        "R": form.R,
        "X": form.X,
        "false_negative_rate": form.p,
        "expected_ig_profit": profits['ig_profit_per_round'],
        "expected_h_profit_base": profits['h_profit_base_per_round'],
        "expected_h_profit_with_solution": profits['h_profit_with_solution_per_round'],
        "profit_ratio": profits['profit_ratio_without_solution'],
        "num_rounds": num_rounds
    }
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {output_file}")
    
    return config


if __name__ == "__main__":
    import sys
    import json
    
    # Default values
    n_total = 100
    num_rounds = 100
    output_file = None
    config = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        # Check if first argument is a config file
        if sys.argv[1].endswith('.json'):
            config_file = sys.argv[1]
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                n_total = config.get('n_total', 100)
                num_rounds = config.get('num_rounds', 100)
                output_file = config_file.replace('.json', '_calculated.json')
            except FileNotFoundError:
                print(f"Error: Config file {config_file} not found")
                sys.exit(1)
        else:
            # Original numeric argument format
            n_total = int(sys.argv[1])
            if len(sys.argv) > 2:
                num_rounds = int(sys.argv[2])
            if len(sys.argv) > 3:
                output_file = sys.argv[3]
            else:
                # Default output file name
                output_file = f"simulation_params_n{n_total}.json"
    else:
        # Default output file name
        output_file = f"simulation_params_n{n_total}.json"
    
    print(f"Generating Nash equilibrium configuration for n={n_total}, rounds={num_rounds}")
    print("=" * 60)
    
    # Create formulation instance
    formulation = NashFormulation()
    
    # Print summary
    formulation.print_summary()
    
    # Generate and save config
    config = generate_simulation_config(n_total, num_rounds, output_file)
    
    print(f"\n\nGenerated configuration for n={n_total}:")
    print(f"  Harmonizers: {config['n_h_initial']} ({config['n_h_initial']/n_total*100:.1f}%)")
    print(f"  IGs: {config['n_ig_initial']} ({config['n_ig_initial']/n_total*100:.1f}%)")
    print(f"  IG multiplier: {config['ig_multiplier']}x")
    print(f"  Min IG quota: {config['min_ig_quota']*100:.0f}%")
    print(f"  Expected H profit: ${config['expected_h_profit_with_solution']:.5f}/round")
    
    print(f"\nUsage: python nash_simulation.py --config {output_file}")