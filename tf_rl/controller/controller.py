class BaseController(object):
    def __init__(self):
        pass

    def action(self, observation):
        raise NotImplementedError('Controller action() not implemented')

    def store(self, observation, action, reward, newobservation):
        raise NotImplementedError('Controller store() not implemented')

    def training_step(self):
        raise NotImplementedError('Controller training_step() not implemented')