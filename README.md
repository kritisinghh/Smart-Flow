# 🚦 Smart Flow: Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning

Smart Flow is an AI-powered traffic control system that uses Multi-Agent Deep Deterministic Policy Gradient (MADDPG) to optimize traffic light timing across multiple intersections. The aim is to reduce average travel time, vehicle waiting time, CO₂ emissions, and fuel consumption in urban environments.

It leverages the SUMO (Simulation of Urban MObility) simulator and a custom reinforcement learning environment to train agents that make intelligent decisions in real-time.

---

# Project Goals

- Reduce congestion at intersections using Reinforcement Learning.
- Optimize environmental and energy metrics: CO₂ emissions and fuel usage.
- Provide a simulation-based testing environment via SUMO.
- Build a scalable, multi-agent system applicable to smart city traffic networks.

---

# Features

 Multi-agent reinforcement learning (MADDPG)  
 SUMO integration for realistic traffic simulation  
 Visual GUI for traffic control replay  
 Metrics for travel time, wait time, CO₂, and fuel usage  
 Support for 2-agent intersection control (extendable)

---

# Technologies Used

| Component              | Technology                            |
|------------------------|---------------------------------------|
| Simulator              | [SUMO](https://www.eclipse.dev/sumo/) |
| RL Algorithm           | MADDPG (Multi-Agent DDPG)             |
| Programming Language   | Python                                |
| Deep Learning          | PyTorch                               |
| Parsing                | XML (ElementTree)                     |
| Visualization          | SUMO-GUI                              |

