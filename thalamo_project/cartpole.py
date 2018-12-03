
import nengo
import numpy as np
from nengo_gym import GymEnv

model = nengo.Network(seed=13)

# x, dx, th, dth
control_matrix = np.array(
    [[-1, 1, -1, 1]]
)

def control(x):

    # x, dx, th, dth

    return - x[0] - x[2] + x[1] + x[3]

with model:

    # dt of CartPole is 0.02
    # dt of Nengo is 0.001
    env = GymEnv(
        env_name='CartPole-v1',
        reset_signal=False,
        reset_when_done=True,
        return_reward=True,
        return_done=False,
        render=True,
        nengo_steps_per_update=20,
    )

    env_node = nengo.Node(
        env,
        size_in=env.size_in,
        size_out=env.size_out
    )

    action = nengo.Ensemble(n_neurons=100, dimensions=1)

    observation = nengo.Ensemble(n_neurons=env.obs_dim*50, dimensions=env.obs_dim)

    reward = nengo.Ensemble(n_neurons=50, dimensions=1)

    nengo.Connection(action, env_node)
    nengo.Connection(env_node[:env.obs_dim], observation)
    nengo.Connection(env_node[env.obs_dim], reward)

    #nengo.Connection(observation, action, transform=control_matrix)

    # Learning with PES rule
    conn = nengo.Connection(
        observation,
        action,
        function=lambda x: [0],
        learning_rule_type=nengo.PES(learning_rate=1e-4),
    )

    nengo.Connection(
        observation,
        conn.learning_rule,
        transform=-1*control_matrix
    )


def on_close(sim):
    env.close()


if __name__ == '__main__':

    sim = nengo.Simulator(model)
    sim.run(10)
