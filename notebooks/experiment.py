from tf_rl.controller import KerasDDPG
from tf_rl.models import DDPGPolicyMLP, DDPGValueMLP
from tf_rl.simulation import DoublePendulum2
from tf_rl import simulate

DOUBLE_PENDULUM_PARAMS = {
    'g_ms2': 9.8,  # acceleration due to gravity, in m/s^2
    'l1_m': 1.0,   # length of pendulum 1 in m
    'm1_kg': 1.0,  # mass of pendulum 1 in kg
    'l2_m': 1.0,
    'm2_kg': 1.0,
    'damping': 0.2,
    'max_control_input': 10.0
}

FPS, SPEED, RES = 60, 1., 0.01


def main():
    actor = DDPGPolicyMLP(DoublePendulum2.observation_size, [200, 200, 1], ['relu', 'relu', 'tanh'])
    critic = DDPGValueMLP(DoublePendulum2.observation_size, DoublePendulum2.action_size, [200, 200, 1], ['relu', 'relu', 'linear'])

    current_controller = KerasDDPG(DoublePendulum2.observation_size, DoublePendulum2.action_size, actor, critic, discount_rate=0.99, exploration_period=1000000)
    try:
        while True:
            d = DoublePendulum2(DOUBLE_PENDULUM_PARAMS)
            print 'Simulating....'
            simulate(d, current_controller, fps=FPS, simulation_resolution=RES, action_every=3, disable_training=False, reset_every=600, visualize=False)
    except KeyboardInterrupt:
        print("Interrupted")

if __name__ == "__main__":
    main()
