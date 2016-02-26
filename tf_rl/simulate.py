import math
import time

from IPython.display import clear_output, display, HTML
from itertools import count
from os.path import join, exists
from os import makedirs

import numpy as np

import matplotlib.pyplot as plt

LOG_FILE_DIR = '/home/mderry/tensorflow-deepq/notebooks/logs/pendulum_'
FILE_EXT = '.png'


def simulate(simulation,
             controller=None,
             fps=60,
             visualize_every=1,
             action_every=1,
             simulation_resolution=None,
             wait=False,
             disable_training=False,
             ignore_exploration=False,
             max_frames=None,
             save_path=None,
             reset_every=None,
             visualize=True):
    """Start the simulation. Performs three tasks

        - visualizes simulation in iPython notebook
        - advances simulator state
        - reports state to controller and chooses actions
          to be performed.

    Parameters
    -------
    simulation: tr_lr.simulation
        simulation that will be simulated ;-)
    controller: tr_lr.controller
        controller used
    fps: int
        frames per seconds
    visualize_every: int
        visualize every `visualize_every`-th frame.
    action_every: int
        take action every `action_every`-th frame
    simulation_resolution: float
        simulate at most 'simulation_resolution' seconds at a time.
        If None, the it is set to 1/FPS (default).
    wait: boolean
        whether to intentionally slow down the simulation
        to appear real time.
    disable_training: bool
        if true training_step is never called.
    max_frames: float
        stop simulation after this many frames.
        (default: never stop)
    save_path: str
        save svg visualization (only tl_rl.utils.svg
        supported for the moment)
    """

    # prepare path to save simulation images
    if save_path is not None:
        if not exists(save_path):
            makedirs(save_path)
    last_image = 0

    # calculate simulation times
    chunks_per_frame = 1
    chunk_length_s   = 1.0 / fps

    if simulation_resolution is not None:
        frame_length_s = 1.0 / fps
        chunks_per_frame = int(math.ceil(frame_length_s / simulation_resolution))
        chunks_per_frame = max(chunks_per_frame, 1)
        chunk_length_s = frame_length_s / chunks_per_frame

    # state transition bookkeeping
    last_observation = None
    last_action      = None

    simulation_started_time = time.time()

    frame_iterator = count() if max_frames is None else range(max_frames)

    reward_history = np.array([])
    reward_mavg_50 = np.array([])

    for frame_no in frame_iterator:
        for _ in range(chunks_per_frame):
            simulation.step(chunk_length_s)

        if frame_no % action_every == 0:
            new_observation = simulation.observe()
            reward = simulation.collect_reward()
            reward_history = np.append(reward_history, [reward])
            if len(reward_history) <= 50:
                # print 'Sub-50, mean: %f' % (np.mean(reward_history))
                reward_mavg_50 = np.append(reward_mavg_50, [np.mean(reward_history)])
            else:
                # print 'Post-50, mean: %f' % (np.mean(reward_history[-50:]))
                reward_mavg_50 = np.append(reward_mavg_50, [np.mean(reward_history[-50:])])

            # print reward_mavg_50.shape

            # store last transition
            if last_observation is not None:
                controller.store(last_observation, last_action, reward, new_observation)

            # act
            new_action = controller.action(new_observation, action_every * chunk_length_s, ignore_exploration=ignore_exploration)
            simulation.perform_action(new_action)

            #train
            if not disable_training:
                # clear_output(wait=True)
                controller.training_step()


            # update current state as last state.
            last_action = new_action
            last_observation = new_observation

        # adding 1 to make it less likely to happen at the same time as
        # action taking.
        if visualize:
            if (frame_no + 1) % visualize_every == 0:
                fps_estimate = frame_no / (time.time() - simulation_started_time)
                clear_output(wait=True)
                svg_html = simulation.to_html(["fps = %.1f" % (fps_estimate,)])
                display(svg_html)
                if save_path is not None:
                    img_path = join(save_path, "%d.svg" % (last_image,))
                    with open(img_path, "w") as f:
                        svg_html.write_svg(f)
                    last_image += 1

        if frame_no % 1000 == 1:
            plot_avg_reward(reward_mavg_50, save=True, filename='%s' % (LOG_FILE_DIR + 'avg_reward_' + str(frame_no) + FILE_EXT))

        time_should_have_passed = frame_no / fps
        time_passed = (time.time() - simulation_started_time)
        if wait and (time_should_have_passed > time_passed):
            time.sleep(time_should_have_passed - time_passed)

        # if reset_every is not None:
        #     simulation.reset()
        #     frame_iterator


def plot_avg_reward(reward_mavg_50, save=False,  filename=None):
    fig = plt.figure()
    plt.plot(reward_mavg_50)
    plt.ylabel('50-window moving average Reward')
    plt.xlabel('Time Step')
    if save:
        plt.savefig(filename, dpi=600)
    else:
        plt.show()
