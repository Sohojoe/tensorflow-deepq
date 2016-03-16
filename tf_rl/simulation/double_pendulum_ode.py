import math
import numpy as np
from scipy.integrate import odeint

import tf_rl.utils.svg as svg
from tf_rl.simulation.simulation import BaseSimulation

g = -9.8


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


def ext_kick(b1, b2, control_input, damping):
    """ This calculates the acceleration on each bob."""
    m1 = b1.m
    l1 = b1.l
    v1 = b1.v
    v12 = v1*v1
    t1 = b1.theta
    m2 = b2.m
    l2 = b2.l
    v2 = b2.v
    v22 = v2*v2
    t2 = b2.theta
    c = np.cos(t1 - t2)
    c1 = np.cos(t1)
    c2 = np.cos(t2)
    s = np.sin(t1 - t2)
    s1 = np.sin(t1)
    s2 = np.sin(t2)
    ss = s * s
    norm = (m1 + m2 * ss)
    a1 = -.5 * m2 * np.sin(2*(t1-t2)) *v12 / norm - l2 * m2 * s * v22 / (l1 * norm)
    a1 += (-.5 * g / l1) * ((2 * m1 + m2) * s1 + m2 * np.sin(t1 - 2 * t2)) / norm
    a1 += control_input
    a1 -= damping * v1
    a2 = l1 * (m1 + m2) * v12 * s / (l2 * norm) + m2 * np.sin(2 * (t1 - t2)) * v22 / (2 * norm)
    a2 += (g / l2) * (m1 + m2) * c1 * s / norm
    a2 -= damping * v2
    return a1, a2


def ext_kick_wrapper(y0, t, b1, b2, control_input, damping):
    b1.theta, b2.theta, b1.v, b2.v = y0
    a1, a2 = ext_kick(b1, b2, control_input, damping)
    res = np.array([b1.v, b2.v, a1, a2])
    return res


class DoublePendulum2(BaseSimulation):
    observation_size = 4
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
        BaseSimulation.__init__(self)
        self.control_input = 0.0
        self.params = params
        self.size = (400, 300)
        self.b1 = Bob(self.params['l1_m'], self.params['m1_kg'], 0.1)
        self.b2 = Bob(self.params['l2_m'], self.params['m2_kg'], 0.0)
        self.damping = self.params['damping']
        junk = self.get_positions()
        junk = self.get_energies()
        self.b1.a, self.b2.a = self.kick()

    def step(self, dt):
        self.pyode(dt)
        x1,y1,x2,y2 = self.get_positions()
        e1,e2 = self.get_energies()

    def reset(self):
        self.b1 = Bob(self.params['l1_m'], self.params['m1_kg'], 0.01)
        self.b2 = Bob(self.params['l2_m'], self.params['m2_kg'], 0.0)
        junk = self.get_positions()
        junk = self.get_energies()
        self.b1.a, self.b2.a = self.kick()

    def kick(self):
        """ This calculates the acceleration on each bob."""
        m1 = self.b1.m
        l1 = self.b1.l
        v1 = self.b1.v
        v12 = v1*v1
        t1 = self.b1.theta
        m2 = self.b2.m
        l2 = self.b2.l
        v2 = self.b2.v
        v22 = v2*v2
        t2 = self.b2.theta
        c = np.cos(t1 - t2)
        c1 = np.cos(t1)
        c2 = np.cos(t2)
        s = np.sin(t1 - t2)
        s1 = np.sin(t1)
        s2 = np.sin(t2)
        ss = s * s
        norm = (m1 + m2 * ss)
        a1 = -.5 * m2 * np.sin(2*(t1-t2)) *v12 / norm - l2 * m2 * s * v22 / (l1 * norm)
        a1 += (-.5 * g / l1) * ((2 * m1 + m2) * s1 + m2 * np.sin(t1 - 2 * t2)) / norm
        a2 = l1 * (m1 + m2) * v12 * s / (l2 * norm) + m2 * np.sin(2 * (t1 - t2)) * v22 / (2 * norm)
        a2 += (g / l2) * (m1 + m2) * c1 * s / norm
        return a1, a2

    def get_positions(self):
        """ Calculate the x,y positions of each bob. """
        l1 = self.b1.l
        t1 = self.b1.theta
        l2 = self.b2.l
        t2 = self.b2.theta
        x1 = l1*np.sin(t1)
        y1 = -l1*np.cos(t1)
        x2 = x1 + l2*np.sin(t2)
        y2 = y1 - l2*np.cos(t2)
        self.b1.x = x1
        self.b1.y = y1
        self.b2.x = x2
        self.b2.y = y2
        return x1, y1, x2, y2

    def get_energies(self):
        """ Calculate the kinetic and potential energies of each bob."""
        x1, y1, x2, y2 = self.get_positions()
        vx1 = -y1*self.b1.v
        vy1 = x1*self.b1.v
        vx2 = vx1 + (y1-y2)*self.b2.v
        vy2 = vy1 + (x2-x1)*self.b2.v
        self.b1.ke = .5*self.b1.m*(vx1**2 + vy1**2)
        self.b1.pe = self.b1.m*g*y1
        self.b1.energy = self.b1.ke + self.b1.pe
        self.b2.ke = .5*self.b2.m*(vx2**2 + vy2**2)
        self.b2.pe = self.b2.m*g*y2
        self.b2.energy = self.b2.ke + self.b2.pe
        return self.b1.energy, self.b2.energy

    def kick_wrapper(self, y0, t):
        """ This is a wrapper to the kick function"""
        a1, a2 = self.kick()
        res = np.array([self.b1.v, self.b2.v, a1, a2])
        return res

    def pyode(self,dt):
        """ This is a wrapper to the odeint integrator. """
        y0 = np.array([self.b1.theta, self.b2.theta, self.b1.v, self.b2.v])
        res = odeint(ext_kick_wrapper, y0, [0, dt], args=(self.b1, self.b2, self.control_input, self.damping))
        self.b1.theta, self.b2.theta, self.b1.v, self.b2.v = res[1]
        if self.b1.theta > np.pi:
            while self.b1.theta > np.pi:
                self.b1.theta -= 2*np.pi
        if self.b1.theta < -np.pi:
            while self.b1.theta < -np.pi:
                self.b1.theta += 2*np.pi
        if self.b2.theta > np.pi:
            while self.b2.theta > np.pi:
                self.b2.theta -= 2*np.pi
        if self.b2.theta < -np.pi:
            while self.b2.theta < -np.pi:
                self.b2.theta += 2*np.pi
        return

    def observe(self):
        return np.matrix([self.b1.theta, self.b1.v, self.b2.theta, self.b2.v])

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
        angle_dist += math.fabs(self.wrap_angle((target_state[2]-current_state[0,2])))
        # angle_dist = 0.0
        vel_dist = np.linalg.norm(target_state[1]-current_state[0,1])
        vel_dist += np.linalg.norm(target_state[3]-current_state[0,3])
        return angle_dist + (0.01*vel_dist)

    def collect_reward(self):
        """Reward corresponds to how high is the first joint."""
        target_state = np.array([0.0, 0.0, 0.0, 0.0])
        current_state = np.matrix([self.b1.theta, self.b1.v, self.b2.theta, self.b2.v])
        distance = self.distance(target_state, current_state)
        action_cost = -0.2 * math.fabs(self.control_input)
        return (2*math.pi + 4.4 - distance) + action_cost
        # _, (x,y) = self.joint_positions()
        # total_length = self.params['l1_m'] + self.params['l2_m']
        # target_x, target_y = 0, -total_length
        # distance_to_target = math.sqrt((x-target_x)**2 + (y-target_y)**2)
        # #abs_vel = abs(self.state[0,1]) + abs(self.state[0,3])
        # #vel_score = math.exp(-abs_vel*5) / 5.0
        # action_cost = -0.01
        # return -distance_to_target / (2.0 * total_length) + action_cost

    def to_html(self, info=[]):
        """Visualize"""
        info = info[:]
        info.append("Reward = %.1f" % self.collect_reward())
        info.append("Joint 1 Angle = %.1f" % self.b1.theta)
        info.append("Joint 1 Velo  = %.1f" % self.b1.v)
        info.append("Joint 2 Angle = %.1f" % self.b2.theta)
        info.append("Joint 2 Velo  = %.1f" % self.b2.v)
        info.append("Control Input = %.1f" % self.control_input)
        junk = self.get_positions()
        joint1 = (self.b1.x, self.b1.y)
        joint2 = (self.b2.x, self.b2.y)

        total_length = self.params['l1_m'] + self.params['l2_m']
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

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 +  20 * len(info)))
        scene.add(svg.Rectangle((10, 10), self.size))

        joint1 = transform(joint1)
        joint2 = transform(joint2)
        scene.add(svg.Line(center, joint1))
        scene.add(svg.Line(joint1, joint2))

        scene.add(svg.Circle(center, 5,  color='red'))
        scene.add(svg.Circle(joint1, 3,  color='blue'))
        scene.add(svg.Circle(joint2, 3,  color='green'))

        offset = self.size[1] + 15
        for txt in info:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20
        return scene
