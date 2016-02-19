import math
import numpy as np

import tf_rl.utils.svg as svg


class DoublePendulum(object):
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
        self.state = np.array([3.1415, 0.0, 3.1415, 0.0])
        self.state = self.state.reshape((1,4))
        self.control_input = 0.0
        self.params = params
        self.size = (400, 300)

    def external_derivatives(self):
        """How state of the world changes
        naturally due to gravity and momentum

        Returns a vector of four values
        which are derivatives of different
        state in the internal state representation."""
        # code below is horrible, if somebody
        # wants to clean it up, I will gladly
        # accept pull request.
        G = self.params['g_ms2']
        L1 = self.params['l1_m']
        L2 = self.params['l2_m']
        M1 = self.params['m1_kg']
        M2 = self.params['m2_kg']
        damping = self.params['damping']
        state = self.state

        dydx = np.zeros_like(state)
        dydx[0,0] = state[0,1]

        del_ = state[0,2]-state[0,0]
        den1 = (M1+M2)*L1 - M2*L1*np.cos(del_)*np.cos(del_)
        dydx[0,1] = (M2*L1*state[0,1]*state[0,1]*np.sin(del_)*np.cos(del_)
                  + M2*G*np.sin(state[0,2])*np.cos(del_)
                  + M2*L2*state[0,3]*state[0,3]*np.sin(del_)
                  - (M1+M2)*G*np.sin(state[0,0])) / den1
        dydx[0,1] -= damping * state[0,1]

        dydx[0,2] = state[0,3]

        den2 = (L2/L1)*den1
        dydx[0,3] = (-M2*L2*state[0,3]*state[0,3]*np.sin(del_)*np.cos(del_)
                   + (M1+M2)*G*np.sin(state[0,0])*np.cos(del_)
                   - (M1+M2)*L1*state[0,1]*state[0,1]*np.sin(del_)
                   - (M1+M2)*G*np.sin(state[0,2]))/den2
        dydx[0,3] -= damping * state[0,3]

        return np.array(dydx)

    def reset(self):
        self.state = np.array([3.1415, 0.0, 3.1415, 0.0])

    def control_derivative(self):
        """Derivative of self.state due to control"""
        return np.array([0., 0., 0., 1.]) * self.control_input

    def observe(self):
        """Returns an observation."""
        return self.state

    def perform_action(self, action):
        """Expects action to be in range [-1, 1]"""
        self.control_input = action * self.params['max_control_input']

    def step(self, dt):
        """Advance simulation by dt seconds"""
        dstate = self.external_derivatives() + self.control_derivative()
        self.state += dt * dstate
        self.state[0,0] = self.wrap_angle(self.state[0,0])
        self.state[0,2] = self.wrap_angle(self.state[0,2])

    def wrap_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def distance(self, target_state, current_state):
        angle_dist = math.fabs(self.wrap_angle((target_state[0]-current_state[0,0])))
        angle_dist += math.fabs(self.wrap_angle((target_state[2]-current_state[0,2])))
        vel_dist = np.linalg.norm(target_state[1]-current_state[0,1])
        vel_dist += np.linalg.norm(target_state[3]-current_state[0,3])
        return vel_dist + angle_dist

    def collect_reward(self):
        """Reward corresponds to how high is the first joint."""
        # target_state = np.array([3.141592, 0, 3.141592, 0])
        # distance = self.distance(target_state, self.state)
        action_cost = -0.01 * math.fabs(self.control_input)
        # return 100 - distance + action_cost
        _, (x,y) = self.joint_positions()
        total_length = self.params['l1_m'] + self.params['l2_m']
        target_x, target_y = 0, -total_length
        distance_to_target = math.sqrt((x-target_x)**2 + (y-target_y)**2)
        # #abs_vel = abs(self.state[0,1]) + abs(self.state[0,3])
        # #vel_score = math.exp(-abs_vel*5) / 5.0
        # action_cost = -0.01
        return -distance_to_target / (2.0 * total_length) + action_cost

    def joint_positions(self):
        """Returns abosolute positions of both joints in coordinate system
        where center of system is the attachement point"""
        x1 = self.params['l1_m']  * np.sin(self.state[0,0])
        y1 = self.params['l1_m'] * np.cos(self.state[0,0])

        x2 = self.params['l2_m']  * np.sin(self.state[0,2]) + x1
        y2 = self.params['l2_m'] * np.cos(self.state[0,2]) + y1

        return (x1, y1), (x2, y2)

    def to_html(self, info=[]):
        """Visualize"""
        info = info[:]
        info.append("Reward = %.1f" % self.collect_reward())
        info.append("Joint 1 Angle = %.1f" % self.state[0,0])
        info.append("Joint 1 Velo  = %.1f" % self.state[0,1])
        info.append("Joint 2 Angle = %.1f" % self.state[0,2])
        info.append("Joint 2 Velo  = %.1f" % self.state[0,3])
        info.append("Control Input = %.1f" % self.control_input)
        joint1, joint2 = self.joint_positions()

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
