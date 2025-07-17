#!/usr/bin/env python3
"""
Reputation-Based Validator Emergence Example

This example demonstrates a blockchain network where:
1. All participants start as Insight Generators (IGs)
2. After 100 correct contributions, IGs earn the right to become validators (H)
3. Only a fraction of eligible IGs choose to validate due to costs/rewards
4. The system reaches a natural equilibrium based on economic incentives
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from multi_player_nash_solver import MultiPlayerParameters, MultiPlayerNashSolver
from nash_equilibrium_solver import Objective, SystemParameters, NashEquilibriumSolver


@dataclass
class NetworkState:
    """Tracks the state of the network at a point in time"""
    hour: int
    day: float  # Calculated from hours
    total_igs: int
    eligible_validators: int  # IGs with 100+ reputation
    active_validators: int    # Eligible IGs who choose to validate
    new_validators_today: int
    ig_reward: float
    h_reward: float
    ig_stake: float
    h_stake: float
    network_cost_daily: float
    security_value: float


class ReputationBasedNetwork:
    """Models a network with reputation-based validator emergence"""
    
    def __init__(self, 
                 initial_igs: int = 100_000,
                 reputation_threshold: int = 100,
                 rounds_per_hour: int = 60,
                 correct_rate: float = 0.95):
        self.initial_igs = initial_igs
        self.reputation_threshold = reputation_threshold
        self.rounds_per_hour = rounds_per_hour
        self.rounds_per_day = rounds_per_hour * 24
        self.correct_rate = correct_rate
        
        # Network growth parameters (converted to hourly)
        self.hourly_ig_growth_rate = 0.01 / 24  # ~0.04% hourly growth
        self.hourly_validator_attrition = 0.001 / 24  # ~0.004% hourly attrition
        
        # Economic parameters that scale with network
        self.base_ig_cost = 0.0004  # Ultra-low IG cost
        self.base_h_cost = 0.04     # Ultra-low H cost
        self.scale_factor = 0.9     # Costs reduce with scale
        
        # Participation model
        self.max_validator_participation = 0.30  # Max 30% of eligible become validators
        
        # Track network history
        self.history: List[NetworkState] = []
        
    def hours_to_validator(self) -> float:
        """Calculate average hours to earn validator status"""
        contributions_per_hour = self.rounds_per_hour * self.correct_rate
        return self.reputation_threshold / contributions_per_hour
    
    def days_to_validator(self) -> float:
        """Calculate average days to earn validator status"""
        return self.hours_to_validator() / 24
    
    def get_costs_at_scale(self, total_participants: int) -> Tuple[float, float]:
        """Calculate costs based on network scale"""
        # Adjust scale calculation for smaller networks
        scale_discount = self.scale_factor ** (np.log10(max(1, total_participants / 1_000)))
        ig_cost = self.base_ig_cost * scale_discount
        h_cost = self.base_h_cost * scale_discount
        return ig_cost, h_cost
    
    def calculate_participation_rate(self, ig_profit: float, h_profit: float) -> float:
        """Calculate what fraction of eligible IGs will become validators"""
        if h_profit <= 0:
            return 0.0
        
        # Sigmoid function based on profitability ratio
        profit_ratio = h_profit / (ig_profit + 0.0001)  # Avoid division by zero
        participation = self.max_validator_participation / (1 + np.exp(-2 * (profit_ratio - 2)))
        return min(participation, self.max_validator_participation)
    
    def simulate_hour(self, hour: int, current_igs: int, current_validators: int) -> NetworkState:
        """Simulate one hour of network activity"""
        # Network growth
        new_igs = int(current_igs * self.hourly_ig_growth_rate)
        total_igs = current_igs + new_igs
        
        # Calculate eligible validators (those with 100+ reputation)
        # Assume linear accumulation over time
        network_age_hours = hour
        avg_hours_to_qualify = self.hours_to_validator()
        
        if network_age_hours < avg_hours_to_qualify:
            # Early stage - very few qualified
            eligible_fraction = (network_age_hours / avg_hours_to_qualify) * 0.01
        else:
            # Mature stage - logarithmic growth
            eligible_fraction = min(0.5, 0.01 + 0.49 * np.log10(network_age_hours / avg_hours_to_qualify) / 2)
        
        eligible_validators = int(total_igs * eligible_fraction)
        
        # Get current costs
        ig_cost, h_cost = self.get_costs_at_scale(total_igs)
        
        # Solve for optimal rewards
        # For 1000 participants, we want ~100 validators (10%)
        # Bootstrap with a reasonable number
        min_validators = 100  # Start with 100 validators in bootstrap
        
        if current_validators < min_validators:
            # Bootstrap phase - need manual intervention
            active_validators = min_validators  # Start with more validators
            ig_reward = ig_cost * 1.3  # 30% margin for IGs
            h_reward = h_cost * 1.4    # 40% margin for validators
            # Much lower stakes: cap at $100 for IGs, $1000 for validators
            ig_stake = min(100, ig_reward * self.rounds_per_day * 1)  # 1 day max
            h_stake = min(1000, h_reward * self.rounds_per_day * 7)  # 7 days max, capped at $1000
        else:
            # Normal operation - use Nash solver
            params = MultiPlayerParameters(
                C_IG_hon=ig_cost,
                C_IG_mal=ig_cost * 0.5,      # Malicious is 50% cheaper
                C_H_hon=h_cost,
                C_H_mal=h_cost * 0.2,        # Malicious validation is 80% cheaper
                n_IG=total_igs,
                n_H=current_validators,
                # For Byzantine tolerance, we need n_H >= 3f+1
                # Select a reasonable subset k for each round (e.g., 5% or at least 3)
                k_H=max(3, current_validators // 20),
                selection_mechanism='quality',
                reward_distribution='proportional',
                p_pass_base=0.02,            # 2% pass rate (slightly higher)
                p_collusion=0.001,
                budget=None
            )
            
            solver = MultiPlayerNashSolver(params)
            solution = solver.solve(Objective.MINIMIZE_BUDGET)
            
            if solution.is_valid():
                # Extract per-participant rewards
                ig_reward = solution.R_IG / total_igs
                # Each validator gets rewarded when selected, with k selections per round
                h_reward = solution.R_H / params.k_H
                
                # Debug: Check Nash equilibrium
                if hour == 24:  # Print at day 1
                    print(f"\n  Nash Equilibrium Check (Hour {hour}):")
                    print(f"    Solution valid: {solution.is_valid()}")
                    print(f"    IG Nash satisfied: {solution.S_IG >= (params.C_IG_hon - params.C_IG_mal)/(1-params.p_pass_base) - ig_reward}")
                    print(f"    H Nash satisfied: {solution.S_H >= (params.C_H_hon - params.C_H_mal)/(1-params.p_pass_base) - h_reward}")
                    print(f"    Computed S_IG: {solution.S_IG:.4f}, S_H: {solution.S_H:.4f}")
                
                # Ensure minimum profit margins (20% for IG, 30% for H)
                min_ig_reward = ig_cost * 1.2
                min_h_reward = h_cost * 1.3
                
                if ig_reward < min_ig_reward:
                    ig_reward = min_ig_reward
                if h_reward < min_h_reward:
                    h_reward = min_h_reward
                
                # Calculate significant stakes based on time commitment
                # IGs: 7 days of rewards, Validators: 14 days of expected rewards
                # CAP at $100 for IGs and $1000 for validators
                ig_stake_calc = max(solution.S_IG, ig_reward * self.rounds_per_day * 7)
                h_stake_calc = max(solution.S_H, h_reward * self.rounds_per_day * 14 * (params.k_H / current_validators))
                
                ig_stake = min(100, ig_stake_calc)
                h_stake = min(1000, h_stake_calc)
            else:
                # Fallback values with profit margins
                ig_reward = ig_cost * 1.2
                h_reward = h_cost * 1.3
                # Significant stakes: 7 days of rewards for IGs, 14 days for validators
                # CAP at $100 for IGs and $1000 for validators
                ig_stake = min(100, ig_reward * self.rounds_per_day * 7)
                h_stake = min(1000, h_reward * self.rounds_per_day * 14 * 0.1)
        
        # Calculate profitability
        ig_profit = ig_reward - ig_cost
        h_profit = h_reward - h_cost
        
        # Determine participation rate
        participation_rate = self.calculate_participation_rate(ig_profit, h_profit)
        target_validators = int(eligible_validators * participation_rate)
        
        # For 1000 participants, we want ~100 validators (10%)
        # Override target if we have enough eligible validators
        if total_igs <= 10000 and eligible_validators >= 100:
            target_validators = max(100, target_validators)
        
        # Validator dynamics
        new_validators = int((target_validators - current_validators) * 0.1 / 24)  # Hourly adjustment
        lost_validators = int(current_validators * self.hourly_validator_attrition)
        active_validators = max(min_validators, current_validators + new_validators - lost_validators)
        
        # Network costs
        total_ig_rewards = ig_reward * total_igs * self.rounds_per_day
        # Calculate k_H for this network size (5% or at least 3)
        k_h = max(3, active_validators // 20)
        total_h_rewards = h_reward * k_h * self.rounds_per_day
        network_cost_daily = total_ig_rewards + total_h_rewards
        
        # Security value
        monetary_security = total_igs * ig_stake + active_validators * h_stake
        reputation_security = eligible_validators * self.reputation_threshold * ig_reward
        security_value = monetary_security + reputation_security
        
        return NetworkState(
            hour=hour,
            day=hour / 24,
            total_igs=total_igs,
            eligible_validators=eligible_validators,
            active_validators=active_validators,
            new_validators_today=new_validators,
            ig_reward=ig_reward,
            h_reward=h_reward,
            ig_stake=ig_stake,
            h_stake=h_stake,
            network_cost_daily=network_cost_daily,
            security_value=security_value
        )
    
    def simulate_network_evolution(self, days: int = 30) -> List[NetworkState]:
        """Simulate network evolution over time (in hours)"""
        hours_to_simulate = days * 24
        current_igs = self.initial_igs
        current_validators = 0
        
        # Define milestone hours (converted from days)
        milestone_hours = [24, 24*7, 24*30]  # 1 day, 1 week, 1 month
        
        for hour in range(hours_to_simulate):
            state = self.simulate_hour(hour, current_igs, current_validators)
            self.history.append(state)
            
            # Update for next iteration
            current_igs = state.total_igs
            current_validators = state.active_validators
            
            # Print milestones
            if hour in milestone_hours:
                self.print_milestone(state)
        
        return self.history
    
    def print_milestone(self, state: NetworkState):
        """Print network state at key milestones"""
        print(f"\nDay {state.day:.1f} Milestone (Hour {state.hour}):")
        print("-" * 50)
        print(f"Network Size:")
        print(f"  Total IGs: {state.total_igs:,}")
        print(f"  Eligible validators: {state.eligible_validators:,}")
        print(f"  Active validators: {state.active_validators:,}")
        print(f"  IG:H ratio: 1:{state.total_igs // max(1, state.active_validators)}")
        
        print(f"\nEconomics:")
        print(f"  IG reward: ${state.ig_reward:.6f}/round")
        print(f"  H reward: ${state.h_reward:.4f}/round")
        print(f"  IG stake: ${state.ig_stake:,.2f} (one-time)")
        print(f"  H stake: ${state.h_stake:,.2f} (one-time)")
        print(f"  Daily network cost: ${state.network_cost_daily:,.2f}")
        print(f"  Annual cost projection: ${state.network_cost_daily * 365:,.2f}")
        
        print(f"\nSecurity:")
        monetary_security = state.total_igs * state.ig_stake + state.active_validators * state.h_stake
        reputation_value = state.security_value - monetary_security
        print(f"  Total stakes locked: ${monetary_security:,.0f}")
        print(f"  Reputation value: ${reputation_value:,.0f}")
        print(f"  Total security value: ${state.security_value:,.0f}")
        print(f"  Security/Cost ratio: {state.security_value / max(1, state.network_cost_daily):.1f}x")
    
    def plot_evolution(self):
        """Create visualizations of network evolution"""
        if not self.history:
            return
        
        days = [s.day for s in self.history]  # Already converted from hours
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Network growth
        ax1.plot(days, [s.total_igs for s in self.history], 'b-', label='Total IGs')
        ax1.plot(days, [s.eligible_validators for s in self.history], 'g-', label='Eligible Validators')
        ax1.plot(days, [s.active_validators for s in self.history], 'r-', label='Active Validators')
        ax1.set_yscale('log')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Participants')
        ax1.set_title('Network Growth Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. IG:H Ratio
        ratios = [s.total_igs / max(1, s.active_validators) for s in self.history]
        ax2.plot(days, ratios, 'purple')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('IG:H Ratio')
        ax2.set_title('Evolution of IG to Validator Ratio')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Economics
        ax3.plot(days, [s.ig_reward * 1440 for s in self.history], 'b-', label='IG Daily Reward')
        ax3.plot(days, [s.h_reward * 1440 * 0.01 for s in self.history], 'r-', label='H Daily Reward (expected)')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Daily Rewards ($)')
        ax3.set_title('Daily Reward Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Security metrics
        ax4.plot(days, [s.security_value for s in self.history], 'g-', label='Security Value')
        ax4.plot(days, [s.network_cost_daily for s in self.history], 'r--', label='Daily Cost')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Value ($)')
        ax4.set_title('Security Value vs Network Cost')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reputation_network_evolution.png', dpi=150)
        print(f"\nVisualization saved to reputation_network_evolution.png")
        
        # Additional analysis plot
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Validator participation rate
        participation_rates = [s.active_validators / max(1, s.eligible_validators) * 100 
                             for s in self.history if s.eligible_validators > 0]
        days_with_eligible = [s.day for s in self.history if s.eligible_validators > 0]
        
        ax5.plot(days_with_eligible, participation_rates, 'orange')
        ax5.set_xlabel('Days')
        ax5.set_ylabel('Participation Rate (%)')
        ax5.set_title('Validator Participation Rate Among Eligible IGs')
        ax5.grid(True, alpha=0.3)
        
        # Cost per participant
        cost_per_ig = [s.network_cost_daily / s.total_igs for s in self.history]
        ax6.plot(days, cost_per_ig, 'brown')
        ax6.set_xlabel('Days')
        ax6.set_ylabel('Daily Cost per IG ($)')
        ax6.set_title('Network Cost Efficiency')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reputation_network_analysis.png', dpi=150)
        print(f"Analysis saved to reputation_network_analysis.png")


def main():
    """Run the reputation-based validator emergence simulation"""
    print("REPUTATION-BASED VALIDATOR EMERGENCE SIMULATION")
    print("=" * 60)
    print("\nNetwork Parameters:")
    print("  Initial IGs: 1,000")
    print("  Reputation threshold: 100 correct contributions")
    print("  Rounds per hour: 60 (1 per minute)")
    print("  Simulation duration: 30 days (720 hours)")
    
    # Create and run simulation
    network = ReputationBasedNetwork(
        initial_igs=1_000,
        reputation_threshold=100,
        rounds_per_hour=60,
        correct_rate=0.95
    )
    
    print(f"\nHours to become validator: {network.hours_to_validator():.1f}")
    print(f"Days to become validator: {network.days_to_validator():.2f}")
    print("\nStarting simulation...")
    
    # Run simulation
    history = network.simulate_network_evolution(days=30)
    
    # Summary statistics
    print("\n\nSIMULATION SUMMARY")
    print("=" * 60)
    
    final_state = history[-1]
    print(f"\nFinal Network State (Day {final_state.day:.1f}):")
    print(f"  Total IGs: {final_state.total_igs:,}")
    print(f"  Active validators: {final_state.active_validators:,}")
    print(f"  Final IG:H ratio: 1:{final_state.total_igs // max(1, final_state.active_validators)}")
    print(f"  Daily network cost: ${final_state.network_cost_daily:,.2f}")
    print(f"  Monthly cost projection: ${final_state.network_cost_daily * 30:,.2f}")
    print(f"  Security value: ${final_state.security_value:,.0f}")
    
    # Growth metrics
    ig_growth = (final_state.total_igs / network.initial_igs - 1) * 100
    validator_growth_start = next((i for i, s in enumerate(history) if s.active_validators > 10), 0)
    days_of_growth = (len(history) - validator_growth_start) / 24
    validator_growth_rate = (final_state.active_validators / history[validator_growth_start].active_validators) ** (1/max(1, days_of_growth)) - 1
    
    print(f"\nGrowth Metrics:")
    print(f"  IG growth: {ig_growth:.1f}% over {final_state.day:.1f} days")
    print(f"  Validator daily growth rate: {validator_growth_rate*100:.2f}%")
    print(f"  Eligible validator pool: {final_state.eligible_validators:,}")
    print(f"  Validator participation: {final_state.active_validators/max(1,final_state.eligible_validators)*100:.1f}%")
    
    # Economic efficiency
    # Use hour 24 (day 1) for initial comparison
    initial_hour = min(24, len(history)-1)
    cost_per_ig_initial = history[initial_hour].network_cost_daily / history[initial_hour].total_igs
    cost_per_ig_final = final_state.network_cost_daily / final_state.total_igs
    efficiency_gain = (1 - cost_per_ig_final / cost_per_ig_initial) * 100
    
    print(f"\nEconomic Efficiency:")
    print(f"  Initial cost per IG/day: ${cost_per_ig_initial:.4f}")
    print(f"  Final cost per IG/day: ${cost_per_ig_final:.4f}")
    print(f"  Efficiency improvement: {efficiency_gain:.1f}%")
    
    # Create visualizations
    network.plot_evolution()
    
    print("\n\nKEY INSIGHTS:")
    print("1. Network grows from 100K to {:,} IGs over {:.1f} days".format(final_state.total_igs, final_state.day))
    print("2. Validator pool emerges organically, reaching {:,} active validators".format(final_state.active_validators))
    print("3. IG:H ratio evolves from bootstrap (1:10) to mature (1:{})".format(
        final_state.total_igs // max(1, final_state.active_validators)))
    print("4. Reputation requirement of {:.1f} hours creates natural sybil resistance".format(network.hours_to_validator()))
    print("5. Economic incentives drive sustainable validator participation")
    print("6. Security value grows to ${:,.0f} through combined stakes and reputation".format(
        final_state.security_value))


if __name__ == "__main__":
    main()