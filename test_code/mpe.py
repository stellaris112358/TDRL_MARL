# %%
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.parallel_env(N=2, render_mode="human", dynamic_rescaling = True, continuous_actions=False)
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # print("observations:", observations)
    print("rewards:", rewards)

env.close()

