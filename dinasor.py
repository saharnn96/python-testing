#!/usr/bin/env python
"""
A simple script to train a DQN agent (using RL) to play a simplified version of the 
“No Internet Dinosaur” (Chrome Dino) game on your PC.

The environment is a minimal simulation where a cactus obstacle moves toward the dinosaur.
The dinosaur may jump (action 1) to avoid collision; doing nothing (action 0) will result 
in a collision if the obstacle is too close and the dino is on (or near) the ground.
"""

import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Define the custom Dino environment
# -------------------------------
class DinoEnv(gym.Env):
    """A very simplified version of the Chrome Dino game."""
    def __init__(self):
        super(DinoEnv, self).__init__()
        # Action space: 0 = do nothing, 1 = jump
        self.action_space = gym.spaces.Discrete(2)
        # Observation space: [dino_y, dino_v, obs_distance]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        # Environment parameters
        self.gravity = 1.0          # Gravity applied each step
        self.jump_velocity = 20.0   # Initial upward velocity when jumping
        self.cactus_height = 30.0   # Height that must be cleared to avoid collision
        self.obstacle_speed = 5.0   # How fast the cactus moves toward the dino
        self.max_steps = 1000       # Maximum steps per episode
        self.reset()

    def reset(self):
        # Reset the dinosaur and obstacle states
        self.dino_y = 0.0         # Dino starts on the ground
        self.dino_v = 0.0         # No vertical velocity initially
        # Place the obstacle at a random distance ahead
        self.obs_distance = np.random.uniform(80, 120)
        self.steps = 0
        self.done = False
        return np.array([self.dino_y, self.dino_v, self.obs_distance], dtype=np.float32)

    def step(self, action):
        """
        Executes one time step in the environment.
          - action: 0 (do nothing) or 1 (jump).
        Returns: observation, reward, done, info
        """
        reward = 1.0  # small reward for surviving this time step
        self.steps += 1

        # Process action: if jump and the dino is on the ground, apply jump impulse.
        if action == 1 and self.dino_y == 0.0:
            self.dino_v = self.jump_velocity

        # Update the dinosaur's vertical position and velocity.
        self.dino_y += self.dino_v
        self.dino_v -= self.gravity
        if self.dino_y < 0:
            self.dino_y = 0
            self.dino_v = 0

        # Move the obstacle (cactus) toward the dinosaur.
        self.obs_distance -= self.obstacle_speed
        # If the cactus has passed the dinosaur, generate a new obstacle.
        if self.obs_distance < -10:
            self.obs_distance = np.random.uniform(80, 120)

        # Collision check:
        # If the cactus is within a “collision window” (here, between -5 and 5 units)
        # and the dino is not high enough (i.e. dino_y < cactus_height), then a collision occurs.
        if -5 < self.obs_distance < 5 and self.dino_y < self.cactus_height:
            self.done = True
            reward = -100  # heavy penalty for collision

        # Optionally, end the episode after a maximum number of steps.
        if self.steps >= self.max_steps:
            self.done = True

        obs = np.array([self.dino_y, self.dino_v, self.obs_distance], dtype=np.float32)
        return obs, reward, self.done, {}

    def render(self, mode='human'):
        # For a simple text render, print out the current state.
        print(f"Step: {self.steps:3d} | Dino Y: {self.dino_y:5.1f} | Dino V: {self.dino_v:5.1f} | Obs Dist: {self.obs_distance:5.1f}")

# -------------------------------
# Define the DQN Agent
# -------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size   = state_size
        self.action_size  = action_size
        self.memory       = deque(maxlen=2000)
        self.gamma        = 0.95    # discount factor
        self.epsilon      = 1.0     # exploration rate (start high)
        self.epsilon_min  = 0.01
        self.epsilon_decay= 0.995
        self.learning_rate= 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Builds a simple neural network to approximate Q(s, a)."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Return an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Predict Q-values for the current state and choose the best action.
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """Train the network using random mini-batches from memory."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Q-learning target: r + gamma * max_a' Q(next_state, a')
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            # Get the current Q-values for the state
            target_f = self.model.predict(np.array([state]), verbose=0)
            # Update the Q-value for the action taken.
            target_f[0][action] = target
            # Train the network for one epoch on this sample.
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        # Slowly reduce the exploration rate.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -------------------------------
# Main training loop
# -------------------------------
if __name__ == "__main__":
    env = DinoEnv()
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            # Uncomment the next line to see a text-based render of the game state.
            # env.render()

            # Choose an action using the agent's policy.
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store the experience in memory.
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # If enough experiences are stored, train the network.
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                print(f"Episode: {e+1:4d}/{episodes} | Score: {total_reward:4.1f} | Epsilon: {agent.epsilon:.2f}")
                break
