import gym
import random
import numpy as np
import json

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

env = gym.make("Ant-v4", render_mode='human')

episodes = 10
best_score = float('-inf')
best_individual = None

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = np.random.uniform(-1, 1, size=8)
        _, reward, done, _ = tuple(env.step(action)[:4])
        score += reward
        env.render()

    print(f"Episode {episode}, Score: {score}")

    if score > best_score:
        best_score = score
        best_individual = action

env.close()

if best_individual is not None:
    data = {"best_individual": best_individual.tolist(), "best_score": best_score}
    with open("best_individual.json", "w") as f:
        json.dump(data, f)
