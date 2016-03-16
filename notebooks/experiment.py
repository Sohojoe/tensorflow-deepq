from tf_rl.controller import KerasDDPG
from tf_rl.models import PolicyMLP, ValueMLP
from tf_rl.simulation import SinglePendulum
from tf_rl import simulate

SINGLE_PENDULUM_PARAMS = {
    'g_ms2': 9.8,  # acceleration due to gravity, in m/s^2
    'l1_m': 1.0,   # length of pendulum 1 in m
    'm1_kg': 1.0,  # mass of pendulum 1 in kg
    'damping': 0.2,
    'max_control_input': 6.0
}

FPS, SPEED, RES = 60, 1., 0.01


def main():
    actor = PolicyMLP(SinglePendulum.observation_size, [200, 200, 1], ['relu', 'relu', 'tanh'])
    critic = ValueMLP(SinglePendulum.observation_size, SinglePendulum.action_size,[200, 200, 1],['relu', 'relu', 'linear'], regularizer=True)

    current_controller = KerasDDPG(SinglePendulum.observation_size, SinglePendulum.action_size, actor, critic, discount_rate=0.99, exploration_period=40000)
    try:
        while True:
            d = SinglePendulum(SINGLE_PENDULUM_PARAMS)
            print 'Simulating....'
            simulate(d, current_controller, fps=FPS, simulation_resolution=RES, action_every=3, disable_training=False, reset_every=600, visualize=False)
    except KeyboardInterrupt:
        print("Interrupted")

if __name__ == "__main__":
    main()
