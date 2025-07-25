

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Update imports to point to the correct directory
sys.path.append(r"C:/Users/kriti/OneDrive/Desktop/capstone")

from env import TrafficEnv
from madddpg import MADDPG
from utils import get_average_travel_time, get_average_CO2, get_average_fuel, get_average_length, get_total_cars

# Argument parser for rendering option
parser = argparse.ArgumentParser()
parser.add_argument("-R", "--render", action="store_true", help="whether render while training or not")
args = parser.parse_args()

if __name__ == "__main__":
    # Check if SUMO_HOME is set
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # Configuration
    state_dim = 10
    action_dim = 2
    n_agents = 2
    n_episode = 70

    # Create Environment and RL Agent
    env = TrafficEnv("gui") if args.render else TrafficEnv()
    agent = MADDPG(n_agents, state_dim, action_dim)

    performance_list = []
    co2_emission = []
    fuel_cons = []
    route_length = []
    cars_list = []
    depart_times = []

    # Create results directory
    results_dir = r"C:/Users/kriti/OneDrive/Desktop/capstone/results"
    os.makedirs(results_dir, exist_ok=True)

    for episode in range(n_episode):
        state = env.reset()
        reward_epi = []
        actions = [None for _ in range(n_agents)]
        action_probs = [None for _ in range(n_agents)]
        done = False

        while not done:
            for i in range(n_agents):
                action, action_prob = agent.select_action(state[i, :], i)
                actions[i] = action
                action_probs[i] = action_prob

            before_state = state
            state, reward, done = env.step(actions)

            transition = [before_state, action_probs, state, reward, done]
            agent.push(transition)

            if agent.train_start():
                for i in range(n_agents):
                    agent.train_model(i)
                agent.update_eps()

            if done:
                break

        env.close()

        # Logging and storing metrics
        avg_travel_time = round(get_average_travel_time(), 2)
        performance_list.append(avg_travel_time)

        avg_length = get_average_length()
        route_length.append(avg_length)

        avg_CO2 = round(get_average_CO2() / avg_length, 2)
        co2_emission.append(avg_CO2)

        avg_fuel = round((get_average_fuel() / avg_length) + 3, 2)
        fuel_cons.append(avg_fuel)

        total_cars = get_total_cars()
        cars_list.append(total_cars)

        depart_times.append(episode * 600)  # Departure time in seconds

        print(
            f"Episode {episode + 1} - Avg Travel Time: {avg_travel_time}s, CO2: {avg_CO2}g/km, Fuel: {avg_fuel}L/100km, Eps: {round(agent.eps, 2)}")

    # Save the model
    model_path = os.path.join(results_dir, f"trained_model{n_episode}.th")
    agent.save_model(model_path)

    # Plot performance metrics
    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 1: CO2 Emission (Line Graph)
    plt.figure(figsize=(10, 5))
    plt.plot(depart_times, co2_emission, label='MADDPG', color='blue')
    plt.xlabel("DEPART (SEC)")
    plt.ylabel("CO2 (G/KM)")
    plt.title("Average CO2 Emission Rate for 1-h Traffic Flow")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "co2_emission.png"))
    plt.show()

    # Plot 2: Fuel Consumption (Bar Graph)
    bar_width = 0.5
    index = np.arange(len(depart_times))

    plt.figure(figsize=(10, 5))
    plt.bar(index, fuel_cons, bar_width, label='MADDPG', color='blue')
    plt.xlabel("DEPART (SEC)")
    plt.ylabel("FUEL (L/100KM)")
    plt.title("Fuel Consumption for 1-h Traffic Flow")
    plt.xticks(index, [str(time) for time in depart_times])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "fuel_consumption.png"))
    plt.show()