class BaseSimulation(object):
    def __init__(self):
        pass

    def observe(self):
        raise NotImplementedError('Simulation observe() not implemented')

    def collect_reward(self):
        raise NotImplementedError('Simulation collect_reward() not implemented')

    def perform_action(self, action):
        raise NotImplementedError('Simulation perform_action() not implemented')

    def step(self, dt):
        raise NotImplementedError('Simulation step() not implemented')

    def reset(self):
        raise NotImplementedError('Simulation reset() not implemented')

    def to_html(self, stats=[]):
        raise NotImplementedError('Simulation to_html() not implemented')
