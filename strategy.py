"""
Strategy template for the Morty Express Challenge.

This file provides a template for implementing your own strategy
to maximize the number of Morties saved.

The challenge: The survival probability of each planet changes over time
(based on the number of trips taken). Your strategy should adapt to these
changing conditions.
"""

from typing_extensions import final
from api_client import SphinxAPIClient
from data_collector import DataCollector
import pandas as pd
import math
from scipy.optimize import curve_fit
import numpy as np
import os

def choose_morties(max_per_trip: int, p_survive: float) -> int:
    """
    Determine number of Morties to send based on estimated survival rate.
    """
    if p_survive >= 0.75:
        return max_per_trip
    elif p_survive >= 0.5:
        return max(2, max_per_trip)
    else:
        return 1

class MortyRescueStrategy:
    """Base class for implementing rescue strategies."""
    
    def __init__(self, client: SphinxAPIClient):
        """
        Initialize the strategy.
        
        Args:
            client: SphinxAPIClient instance
        """
        self.client = client
        self.collector = DataCollector(client)
        self.exploration_data = []
    
    def explore_phase(self, trips_per_planet: int = 30) -> pd.DataFrame:
        """
        Initial exploration phase to understand planet behaviors.
        
        Args:
            trips_per_planet: Number of trips to send to each planet
            
        Returns:
            DataFrame with exploration data
        """
        print("\n=== EXPLORATION PHASE ===")
        df = self.collector.explore_all_planets(
            trips_per_planet=trips_per_planet,
            morty_count=1  # Send 1 Morty at a time during exploration
        )
        self.exploration_data = df
        return df
    
    def analyze_planets(self) -> dict:
        """
        Analyze planet data to determine characteristics.
        
        Returns:
            Dictionary with analysis results
        """
        if len(self.exploration_data) == 0:
            raise ValueError("No exploration data available. Run explore_phase() first.")
        
        return self.collector.analyze_risk_changes(self.exploration_data)
    
    def execute_strategy(self):
        """
        Execute the main rescue strategy.
        Override this method to implement your own strategy.
        """
        raise NotImplementedError("Implement your strategy in a subclass")


class SimpleGreedyStrategy(MortyRescueStrategy):
    """
    Simple greedy strategy: always pick the planet with highest recent success.
    """
    
    def execute_strategy(self, morties_per_trip: int = 3):
        """
        Execute the greedy strategy.
        
        Args:
            morties_per_trip: Number of Morties to send per trip (1-3)
        """
        print("\n=== EXECUTING GREEDY STRATEGY ===")
        
        # Get current status
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        print(f"Starting with {morties_remaining} Morties in Citadel")
        
        # Determine best planet from exploration
        best_planet, best_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        
        print(f"Best planet identified: {best_planet_name}")
        print(f"Sending all remaining Morties to {best_planet_name}...")
        
        trips_made = 0
        
        while morties_remaining > 0:
            # Determine how many to send
            morties_to_send = min(morties_per_trip, morties_remaining)
            
            # Send Morties
            result = self.client.send_morties(best_planet, morties_to_send)
            
            morties_remaining = result['morties_in_citadel']
            trips_made += 1
            
            if trips_made % 50 == 0:
                print(f"  Progress: {trips_made} trips, "
                      f"{result['morties_on_planet_jessica']} saved, "
                      f"{morties_remaining} remaining")
        
        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")


class AdaptiveStrategy(MortyRescueStrategy):
    """
    Adaptive strategy: continuously monitor and switch planets if needed.
    """
    
    def execute_strategy(
        self,
        morties_per_trip: int = 3,
        reevaluate_every: int = 50
    ):
        """
        Execute the adaptive strategy.
        
        Args:
            morties_per_trip: Number of Morties to send per trip (1-3)
            reevaluate_every: Re-evaluate best planet every N trips
        """
        print("\n=== EXECUTING ADAPTIVE STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        print(f"Starting with {morties_remaining} Morties in Citadel")
        
        # Initial best planet
        current_planet, current_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        
        print(f"Starting with planet: {current_planet_name}")
        
        trips_since_evaluation = 0
        total_trips = 0
        recent_results = []
        
        while morties_remaining > 0:
            # Send Morties
            morties_to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(current_planet, morties_to_send)
            
            # Track recent results
            recent_results.append({
                'planet': current_planet,
                'survived': result['survived']
            })
            
            morties_remaining = result['morties_in_citadel']
            trips_since_evaluation += 1
            total_trips += 1
            
            # Re-evaluate strategy periodically
            if trips_since_evaluation >= reevaluate_every and morties_remaining > 0:
                # Check if we should switch planets
                recent_success_rate = sum(
                    r['survived'] for r in recent_results[-reevaluate_every:]
                ) / min(len(recent_results), reevaluate_every)
                
                print(f"\n  Re-evaluating at trip {total_trips}...")
                print(f"  Current planet: {current_planet_name}")
                print(f"  Recent success rate: {recent_success_rate*100:.2f}%")
                
                # TODO: Implement logic to potentially switch planets
                # For now, we stick with the same planet
                
                trips_since_evaluation = 0
            
            if total_trips % 50 == 0:
                print(f"  Progress: {total_trips} trips, "
                      f"{result['morties_on_planet_jessica']} saved")
        
        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")


def run_strategy(strategy_class, explore_trips: int = 30):
    """
    Run a complete strategy from exploration to execution.
    
    Args:
        strategy_class: Strategy class to use
        explore_trips: Number of exploration trips per planet
    """
    # Initialize client and strategy
    client = SphinxAPIClient()
    strategy = strategy_class(client)
    
    # Start new episode
    print("Starting new episode...")
    client.start_episode()
    
    """     # Exploration phase
    strategy.explore_phase(trips_per_planet=explore_trips)
    
    # Analyze results
    analysis = strategy.analyze_planets()
    print("\nPlanet Analysis:")
    for planet_name, data in analysis.items():
        print(f"  {planet_name}: {data['overall_survival_rate']:.2f}% "
              f"({data['trend']})") """
    
    # Execute strategy
    strategy.execute_strategy()

class UCBStrategy(MortyRescueStrategy):
    """UCB strategy with adaptive number of Morties."""
    
    def __init__(self, client, exploration_trips: int = 10):
        super().__init__(client)
        self.exploration_trips = exploration_trips
        self.total_trips = 0
        self.planet_stats = {i: {"trips":0, "morties_sent":0, "morties_saved":0} for i in range(3)}
    
    def _select_ucb_planet(self) -> int:
        scores = {}
        for pid, stats in self.planet_stats.items():
            if stats["trips"] == 0:
                scores[pid] = float("inf")
                continue
            avg_reward = stats["morties_saved"] / max(stats["morties_sent"], 1)
            bonus = math.sqrt(2 * math.log(max(1,self.total_trips)) / stats["trips"])
            scores[pid] = avg_reward + bonus
        return max(scores, key=scores.get)
    
    def execute_strategy(self, morties_per_trip: int = 1):
        print("\n=== EXECUTING UCB STRATEGY ===")
        exploration_df = self.explore_phase(trips_per_planet=self.exploration_trips)
        
        for planet_id, group in exploration_df.groupby("planet"):
            trips = len(group)
            sent = group["morties_sent"].sum()
            saved = group["survived"].sum()
            self.planet_stats[planet_id]["trips"] += trips
            self.planet_stats[planet_id]["morties_sent"] += sent
            self.planet_stats[planet_id]["morties_saved"] += saved
            self.total_trips += trips
        
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        last_saved = status["morties_on_planet_jessica"]
        
        while morties_remaining > 0:
            chosen = self._select_ucb_planet()
            planet_data = self.exploration_data[self.exploration_data['planet'] == chosen]
            p_estimate = planet_data['survived'].mean() if len(planet_data)>0 else 0.5
            to_send = min(choose_morties(morties_per_trip, p_estimate), morties_remaining)
            
            result = self.client.send_morties(chosen, to_send)
            saved_now = max(result["morties_on_planet_jessica"] - last_saved, 0)
            last_saved = result["morties_on_planet_jessica"]
            
            self.planet_stats[chosen]["trips"] += 1
            self.planet_stats[chosen]["morties_sent"] += to_send
            self.planet_stats[chosen]["morties_saved"] += saved_now
            self.total_trips += 1
            morties_remaining = result["morties_in_citadel"]
            
            if self.total_trips % 50 == 0:
                print(f"  Trips: {self.total_trips}, Send: {1000-morties_remaining}, Saved: {last_saved}, Remaining: {morties_remaining}")
        final = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final['morties_lost']}")
        print(f"Total Steps: {final['steps_taken']}")
        print(f"Success Rate: {(final['morties_on_planet_jessica']/1000)*100:.2f}%")


class UCBSlidingWindowStrategy(MortyRescueStrategy):
    """UCB strategy with sliding window and adaptive Morties."""
    
    def __init__(self, client, exploration_trips: int = 10, window_size: int = 100):
        super().__init__(client)
        self.exploration_trips = exploration_trips
        self.window_size = window_size
        self.total_trips = 0
        self.planet_stats = {i: {"recent_trips": []} for i in range(3)}
    
    def _select_ucb_planet(self) -> int:
        scores = {}
        for pid, stats in self.planet_stats.items():
            trips = len(stats["recent_trips"])
            if trips == 0:
                scores[pid] = float("inf")
                continue
            sent = sum(tr["sent"] for tr in stats["recent_trips"])
            saved = sum(tr["saved"] for tr in stats["recent_trips"])
            avg_reward = saved / max(sent, 1)
            bonus = math.sqrt(2 * math.log(max(1,self.total_trips)) / trips)
            scores[pid] = avg_reward + bonus
        return max(scores, key=scores.get)
    
    def _update_planet_stats(self, planet_id: int, sent: int, saved: int):
        recent = self.planet_stats[planet_id]["recent_trips"]
        recent.append({"sent": sent, "saved": saved})
        if len(recent) > self.window_size:
            recent.pop(0)
    
    def execute_strategy(self, morties_per_trip: int = 1):
        print("\n=== EXECUTING UCB SLIDING WINDOW STRATEGY ===")
        exploration_df = self.explore_phase(trips_per_planet=self.exploration_trips)
        
        for planet_id, group in exploration_df.groupby("planet"):
            for _, row in group.iterrows():
                self._update_planet_stats(planet_id, row["morties_sent"], row["survived"])
                self.total_trips += 1
        
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        last_saved = status["morties_on_planet_jessica"]
        
        while morties_remaining > 0:
            chosen = self._select_ucb_planet()
            planet_data = self.exploration_data[self.exploration_data['planet'] == chosen]
            p_estimate = planet_data['survived'].mean() if len(planet_data)>0 else 0.5
            to_send = min(choose_morties(morties_per_trip, p_estimate), morties_remaining)
            
            result = self.client.send_morties(chosen, to_send)
            saved_now = max(result["morties_on_planet_jessica"] - last_saved, 0)
            last_saved = result["morties_on_planet_jessica"]
            
            self._update_planet_stats(chosen, to_send, saved_now)
            self.total_trips += 1
            morties_remaining = result["morties_in_citadel"]
            
            if self.total_trips % 50 == 0:
                print(f"  Trips: {self.total_trips}, Send: {1000-morties_remaining}, Saved: {last_saved}, Remaining: {morties_remaining}")

        final = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final['morties_lost']}")
        print(f"Total Steps: {final['steps_taken']}")
        print(f"Success Rate: {(final['morties_on_planet_jessica']/1000)*100:.2f}%")


class SinusPredictiveStrategy(MortyRescueStrategy):
    """Predict survival peaks using sinusoidal modeling with adaptive Morties."""
    
    def __init__(self, client, exploration_trips=10):
        super().__init__(client)
        self.exploration_trips = exploration_trips
        self.total_trips = 0
        self.planet_history = {i: [] for i in range(3)}
        self.planet_params = {}
    
    @staticmethod
    def _sin_func(x, A, B, phi, C):
        return A * np.sin(B * x + phi) + C
    
    def _fit_sinus(self, planet_id):
        history = self.planet_history[planet_id]
        if len(history) < 5:
            return None
        x = np.array([trip for trip, _ in history])
        y = np.array([survived for _, survived in history])
        guess = [0.5, 2*np.pi/len(x), 0, np.mean(y)]
        try:
            params, _ = curve_fit(self._sin_func, x, y, p0=guess, maxfev=5000)
            return params
        except:
            return None
    
    def _predict_survival(self, planet_id, next_trip):
        if planet_id not in self.planet_params:
            return 0.5
        A, B, phi, C = self.planet_params[planet_id]
        return self._sin_func(next_trip, A, B, phi, C)
    
    def execute_strategy(self, morties_per_trip=2):
        print("\n=== EXECUTING SINUS PREDICTIVE STRATEGY ===")
        exploration_df = self.explore_phase(trips_per_planet=self.exploration_trips)
        
        for planet_id, group in exploration_df.groupby("planet"):
            for _, row in group.iterrows():
                self.planet_history[planet_id].append((row["trip_number"], row["survived"]))
                self.total_trips += 1
        
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        last_saved = status["morties_on_planet_jessica"]
        
        while morties_remaining > 0:
            next_trip = self.total_trips + 1
            for pid in range(3):
                params = self._fit_sinus(pid)
                if params is not None:
                    self.planet_params[pid] = params
            predictions = {pid: self._predict_survival(pid, next_trip) for pid in range(3)}
            chosen = max(predictions, key=predictions.get)
            p_estimate = predictions[chosen]
            to_send = min(choose_morties(morties_per_trip, p_estimate), morties_remaining)
            
            result = self.client.send_morties(chosen, to_send)
            saved_now = max(result["morties_on_planet_jessica"] - last_saved, 0)
            last_saved = result["morties_on_planet_jessica"]
            
            self.planet_history[chosen].append((next_trip, saved_now))
            self.total_trips += 1
            morties_remaining = result["morties_in_citadel"]
            
            if self.total_trips % 50 == 0:
                print(f"  Trips: {self.total_trips}, Send: {1000-morties_remaining}, Saved: {last_saved}, Remaining: {morties_remaining}")
        final = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final['morties_lost']}")
        print(f"Total Steps: {final['steps_taken']}")
        print(f"Success Rate: {(final['morties_on_planet_jessica']/1000)*100:.2f}%")

import matplotlib.pyplot as plt

class ProbeSinglePlanetStrategy(MortyRescueStrategy):

    def __init__(self, client: SphinxAPIClient, target_planet: int = 0, max_trips: int = None):
        super().__init__(client)
        self.target_planet = int(target_planet)
        self.max_trips = max_trips
        self.instant_rates = []
        self.cumulative_rates = []

    def execute_strategy(self, morties_per_trip: int = 1):
        if morties_per_trip != 1:
            print("Warning: Probe strategy works best with 1 Morty per trip.")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        total_survivors = status["morties_on_planet_jessica"]

        trips_done = 0
        limit = self.max_trips if self.max_trips else morties_remaining

        print(f"\n=== PROBING PLANET {self.target_planet} ===")
        print(f"Starting probe with limit = {limit}")

        while morties_remaining > 0 and trips_done < limit:
            result = self.client.send_morties(self.target_planet, 1)

            instant = result["survived"]
            total_survivors += instant
            trips_done += 1
            morties_remaining = result["morties_in_citadel"]

            self.instant_rates.append(instant)
            self.cumulative_rates.append(total_survivors / trips_done)

            if trips_done % 100 == 0:
                print(f" Trip {trips_done}: Instant={instant}, "
                      f"Cumulative={self.cumulative_rates[-1]:.3f}, "
                      f"Remaining={morties_remaining}")

        print("\nProbe finished.")
        self._save_and_plot()

    def _save_and_plot(self):
        trips = list(range(1, len(self.instant_rates) + 1))

        os.makedirs("plots", exist_ok=True)

        # --- Graph 1 : Instant survival ---
        plt.figure(figsize=(12, 4))
        plt.step(trips, self.instant_rates, where="mid")
        plt.title(f"Instant Survival — Planet {self.target_planet}")
        plt.xlabel("Trip Number")
        plt.ylabel("Survival (0/1)")
        plt.grid(True)
        plt.tight_layout()
        file1 = f"plots/instant_survival_planet{self.target_planet}.png"
        plt.savefig(file1)
        plt.show()
        print(f"Saved: {file1}")

        # --- Graph 2 : Cumulative survival ---
        plt.figure(figsize=(12, 4))
        cum_percent = [c * 100 for c in self.cumulative_rates]
        plt.plot(trips, cum_percent)
        plt.title(f"Cumulative Survival Rate — Planet {self.target_planet}")
        plt.xlabel("Trip Number")
        plt.ylabel("Cumulative Survival (%)")
        plt.ylim(-5, 105)
        plt.grid(True)
        plt.tight_layout()
        file2 = f"plots/cumulative_survival_planet{self.target_planet}.png"
        plt.savefig(file2)
        plt.show()
        print(f"Saved: {file2}")


if __name__ == "__main__":
    print("Morty Express Challenge - Strategy Module")
    print("="*60)
    
    print("\nAvailable strategies:")
    print("1. SimpleGreedyStrategy - Pick best planet and stick with it")
    print("2. AdaptiveStrategy - Monitor and adapt to changing conditions")
    
    print("\nExample usage:")
    print("  run_strategy(SimpleGreedyStrategy, explore_trips=30)")
    print("  run_strategy(AdaptiveStrategy, explore_trips=30)")
    
    print("\nTo create your own strategy:")
    print("1. Subclass MortyRescueStrategy")
    print("2. Implement the execute_strategy() method")
    print("3. Use self.client to interact with the API")
    print("4. Use self.collector to analyze data")


    
    # Uncomment to run:
    run_strategy(ProbeSinglePlanetStrategy, explore_trips=10)
