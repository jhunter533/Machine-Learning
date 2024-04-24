import gymnasium as gym
env=gym.make('Pendulum-v1',render_mode='human')
observation=env.reset()
env.render()
for _ in range(100):
    action=env.action_space.sample()
    observation,reward,done,_,info=env.step(action)
    env.render()
    if done:
        observation=env.reset()
env.close()
