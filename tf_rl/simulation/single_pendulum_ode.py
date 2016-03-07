import math
import numpy as np
import random
from scipy.integrate import odeint

import tf_rl.utils.svg as svg

g = 9.8


class Bob():
    """ This class holds all of pendulum variables such as 
    position, velocity, length, mass, and energy.
    """
    def __init__(self,length, mass, initial_angle):
        self.l = length 
        self.m = mass
        self.theta = initial_angle
        self.v = 0
        self.x = 0
        self.y = 0
        self.p = 0 
        self.a = 0
        self.energy = 0
        self.ke = 0
        self.pe = 0


def ext_kick(b1, control_input, damping):
    """ This calculates the acceleration on each bob."""
    l1 = b1.l
    v1 = b1.v
    t1 = b1.theta
    a1 = g/l1 * np.sin(t1)
    a1 += control_input
    a1 -= damping * v1
    return a1


def ext_kick_wrapper(y0, t, b1, control_input, damping):
    b1.theta, b1.v = y0
    a1 = ext_kick(b1, control_input, damping)
    res = np.array([b1.v, a1])
    return res


class SinglePendulum(object):
    observation_size = 2
    action_size      = 1

    def __init__(self, params):
        """Double Pendulum simulation, where control is
        only applied to joint1.

        state of the system is encoded as the following
        four values:
        state[0]:
            angle of first bar from center
            (w.r.t. vertical axis)
        state[1]:
            angular velocity of state[0]
        state[2]:
            angle of second bar from center
            (w.r.t vertical axis)
        state[3]:
            angular velocity of state[2]

        Params
        -------
        g_ms2 : float
            gravity acceleration
        l1_m : float
            length of the first bar (closer to center)
        m1_kg: float
            mass of the first joint
        l2_m : float
            length of the second bar
        m2_kg : float
            mass of the second joint
        max_control_input : float
            maximum value of angular force applied
            to the first joint
        """
        self.control_input = 0.0
        self.params = params
        self.size = (400, 300)
        self.b1 = Bob(self.params['l1_m'], self.params['m1_kg'], np.pi)
        self.damping = self.params['damping']
        junk = self.get_positions()
        junk = self.get_energies()
        self.b1.a = self.kick()

    def step(self, dt):
        self.pyode(dt)
        x1,y1 = self.get_positions()
        e1 = self.get_energies()

    def reset(self):
        # init_angle = random.uniform(-np.pi, np.pi)
        init_angle = np.pi
        self.b1 = Bob(self.params['l1_m'], self.params['m1_kg'], init_angle)
        junk = self.get_positions()
        junk = self.get_energies()
        self.b1.a = self.kick()

    def kick(self):
        """ This calculates the acceleration on each bob."""
        l1 = self.b1.l
        t1 = self.b1.theta
        a1 = g/l1 * np.sin(t1)
        return a1

    def get_positions(self):
        """ Calculate the x,y positions of each bob. """
        l1 = self.b1.l
        t1 = self.b1.theta
        x1 = l1*np.sin(t1)
        y1 = -l1*np.cos(t1)
        self.b1.x = x1
        self.b1.y = y1
        return x1, y1

    def get_energies(self):
        """ Calculate the kinetic and potential energies of each bob."""
        x1, y1 = self.get_positions()
        vx1 = -y1*self.b1.v
        vy1 = x1*self.b1.v
        self.b1.ke = .5*self.b1.m*(vx1**2 + vy1**2)
        self.b1.pe = self.b1.m*g*y1
        self.b1.energy = self.b1.ke + self.b1.pe
        return self.b1.energy

    def kick_wrapper(self, y0, t):
        """ This is a wrapper to the kick function"""
        a1 = self.kick()
        res = np.array([self.b1.v, a1])
        return res

    def pyode(self,dt):
        """ This is a wrapper to the odeint integrator. """
        y0 = np.array([self.b1.theta, self.b1.v])
        res = odeint(ext_kick_wrapper, y0, [0, dt], args=(self.b1, self.control_input, self.damping))
        self.b1.theta, self.b1.v = res[1]
        if self.b1.theta > np.pi:
            while self.b1.theta > np.pi:
                self.b1.theta -= 2*np.pi
        if self.b1.theta < -np.pi:
            while self.b1.theta < -np.pi:
                self.b1.theta += 2*np.pi
        return

    def observe(self):
        return np.matrix([self.b1.theta, self.b1.v])

    def perform_action(self, action):
        """Expects action to be in range [-1, 1]"""
        self.control_input = action * self.params['max_control_input']

    def wrap_angle(self, angle):
        if angle > np.pi:
            while angle > np.pi:
                angle -= 2*np.pi
        if angle < -np.pi:
            while angle < -np.pi:
                angle += 2*np.pi
        return angle

    def distance(self, target_state, current_state):
        angle_dist = math.fabs(self.wrap_angle((target_state[0]-current_state[0,0])))
        vel_dist = math.fabs(target_state[1]-current_state[0,1])
        # vel_dist = 0.0
        return angle_dist + (0.1*vel_dist)

    def collect_reward(self):
        """Reward corresponds to how high is the first joint."""
        target_state = np.array([0.0, 0.0])
        current_state = np.matrix([self.b1.theta, self.b1.v])
        distance = self.distance(target_state, current_state)
        action_cost = -0.5 * math.fabs(self.control_input)
        return (math.pi + 4.4 - distance) + action_cost

    def to_html(self, info=[]):
        """Visualize"""
        info = info[:]
        info.append("Reward = %.1f" % self.collect_reward())
        info.append("Joint 1 Angle = %.1f" % self.b1.theta)
        info.append("Joint 1 Velo  = %.1f" % self.b1.v)
        info.append("Control Input = %.1f" % self.control_input)
        junk = self.get_positions()
        joint1 = (self.b1.x, self.b1.y)

        total_length = self.params['l1_m']
        # 9 / 10 th of half the screen width
        total_length_px = (8./10.) * (min(self.size) / 2.)
        scaling_ratio = total_length_px / total_length
        center = (self.size[0] / 2, self.size[1] / 2)

        def transform(point):
            """Transforms from state reference world
            to screen and pixels reference world"""

            x = center[0] + scaling_ratio * point[0]
            y = center[1] + scaling_ratio * point[1]
            return int(x), int(y)

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(info)))
        scene.add(svg.Rectangle((10, 10), self.size))

        joint1 = transform(joint1)
        scene.add(svg.Line(center, joint1))

        scene.add(svg.Circle(center, 5,  color='red'))
        scene.add(svg.Circle(joint1, 3,  color='blue'))

        offset = self.size[1] + 15
        for txt in info:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20
        return scene
