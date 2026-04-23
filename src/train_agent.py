# src/train_agent.py
from q_learning import QLearningAgent
from environment import AsteroidEnvironment

def get_trained_agent(grid_size=10, num_obstacles=15, alpha=0.1, gamma=0.9, epsilon=0.2, episodes=500):
    """
    Train the agent and return it for use in data collection.
    """
    env = AsteroidEnvironment(grid_size=grid_size, num_obstacles=num_obstacles)
    agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    
    agent.train()  # run training
    return agent

if __name__ == "__main__":
    agent = get_trained_agent()
    agent.test()  # optional visualization
