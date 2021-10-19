import asyncio
import gym
import random

from neat import Neat
from genome import Genome

async def run_gym(genome: Genome, should_render=False):
    env = gym.make("CartPole-v0")
    env.reset()
    fitness = 0.0
    observation, reward, done, info = env.step(env.action_space.sample())
    
    for x in range(1000):
        if should_render:
            env.render()
        result = genome.feed_forward(observation)
        observation, reward, done, info = env.step(result.index(max(result)))
        fitness += reward
        if done:
            print(f"Simulation completed after {x + 1} timesteps")
            break
    env.close()
    
    genome.fitness = fitness

# async def main():
#     neat = Neat(4, 2, 250, run_gym)
#     x = 0
#     for _ in range(20):
#         await neat.create_generation()
    
#     print(f"best genome fitness: {neat.best_genome.fitness}")
#     for connection in neat.best_genome.connections:
#         print(connection)
#     # run_gym(neat.best_genome, should_render=True)
#     print(f"ran {x} times")

xor_outcomes = {
    (1, 1): 0,
    (1, 0): 1,
    (0, 1): 1,
    (0, 0): 0
}

async def xor(genome):
    delta = 0
    for _ in range(4):
        inputs = random.choice(list(xor_outcomes))
        result = genome.feed_forward(inputs)[0]
        delta += abs(result - xor_outcomes[inputs])
    genome.fitness = 1 / (delta if delta != 0 else 0.001)

async def main():
    print(list(xor_outcomes))
    neat = Neat(2, 1, 256, xor)
    for _ in range(500):
        await neat.create_generation()
        if neat.best_genome.fitness > 1000:
            break
    for inputs in xor_outcomes.keys():
        print(f"{inputs}: {neat.best_genome.feed_forward(inputs)}")
    for connection in neat.best_genome.connections:
        print(connection)

if __name__ == '__main__':
    asyncio.run(main())