import argparse
from datetime import datetime
import logging

import random as rand
import numpy as np

import environments
import experiments

from experiments import plotting


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configure rewards per environment
ENV_REWARDS = {
               'small_lake': {
                              'step_rew': -1,
                              'hole_rew': -100,
                              'goal_rew': 100,
                             },
               'large_lake': {
                              'step_rew': -1,
                              'hole_rew': -100,
                              'goal_rew': 100,
                             },
               'cliff_walking': {
                                 'step_rew': -1,
                                 'fall_rew': -100,
                                 'goal_rew': 100,
                                },
              }

# Configure max steps per experiment
MAX_STEPS = {
             'pi': 10000,
             'vi': 200,
             'ql': 30000,
            }

# Configure trials per experiment
NUM_TRIALS = {
             'pi': 2000,
             'vi': 100,
             'ql': 1000,
            }

# Configure thetas per experiment
PI_THETA = 0.00001
VI_THETA = 0.00001
QL_THETA = 0.001

# Configure minimum consecutive sub-theta episodes and max episodes for QL experiment
QL_EPSILON_DECAYS = [0.00001]
QL_MIN_SUB_THETAS = 5
QL_MAX_EPISODES = max(MAX_STEPS['ql'], NUM_TRIALS['ql'], 30000) # 30000


def run_experiment(experiment_details, experiment, timing_key, verbose, timings, max_steps, num_trials, \
                   theta = None, max_episodes = None, min_sub_thetas = None, epsilon_decays = None):

    timings[timing_key] = {}
    for details in experiment_details:
        t = datetime.now()
        logger.info("Running {} experiment: {}".format(timing_key, details.env_readable_name))
        if max_episodes is None:
            exp = experiment(details, verbose=verbose, max_steps=max_steps, num_trials=num_trials, theta=theta)
        else:
            exp = experiment(details, verbose=verbose, max_steps=max_steps, num_trials=num_trials,
                             max_episodes=max_episodes, min_sub_thetas=min_sub_thetas, theta=theta,
                             epsilon_decays=epsilon_decays)
        exp.perform()
        t_d = datetime.now() - t
        timings[timing_key][details.env_name] = t_d.seconds


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run MDP experiments')
    parser.add_argument('--threads', type=int, default=-1, help='Number of threads (defaults to -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--policy', action='store_true', help='Run the Policy Iteration (PI) experiment')
    parser.add_argument('--value', action='store_true', help='Run the Value Iteration (VI) experiment')
    parser.add_argument('--ql', action='store_true', help='Run the Q-Learner (QL) experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--plot', action='store_true', help='Plot data results')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    logger.info("Creating MDPs")
    logger.info("----------")

    envs = [
        {
            'env': environments.get_rewarding_frozen_lake_8x8_environment(ENV_REWARDS['small_lake']['step_rew'],
                                                                          ENV_REWARDS['small_lake']['hole_rew'],
                                                                          ENV_REWARDS['small_lake']['goal_rew']),
            'name': 'frozen_lake',
            'readable_name': 'Frozen Lake (8x8)',
        },
        {
            'env': environments.get_large_rewarding_frozen_lake_15x15_environment(ENV_REWARDS['large_lake']['step_rew'],
                                                                                  ENV_REWARDS['large_lake']['hole_rew'],
                                                                                  ENV_REWARDS['large_lake']['goal_rew']),
            'name': 'large_frozen_lake',
            'readable_name': 'Frozen Lake (15x15)',
        },
        {
            'env': environments.get_windy_cliff_walking_4x12_environment(ENV_REWARDS['cliff_walking']['step_rew'],
                                                                         ENV_REWARDS['cliff_walking']['fall_rew'],
                                                                         ENV_REWARDS['cliff_walking']['goal_rew']),
            'name': 'cliff_walking',
            'readable_name': 'Cliff Walking (4x12)',
        }
    ]

    experiment_details = []
    for env in envs:
        env['env'].seed(seed)
        logger.info('{}: State space: {}, Action space: {}'.format(env['readable_name'], env['env'].unwrapped.nS,
                                                                   env['env'].unwrapped.nA))
        experiment_details.append(experiments.ExperimentDetails(
            env['env'], env['name'], env['readable_name'],
            threads=threads,
            seed=seed
        ))

    if verbose:
        logger.info("----------")
    print('\n\n')
    logger.info("Running experiments")

    timings = {}

    if args.policy or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.PolicyIterationExperiment, 'PI', verbose, timings, \
                       MAX_STEPS['pi'], NUM_TRIALS['pi'], theta=PI_THETA)

    if args.value or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.ValueIterationExperiment, 'VI', verbose, timings, \
                       MAX_STEPS['vi'], NUM_TRIALS['vi'], theta=VI_THETA)

    if args.ql or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.QLearnerExperiment, 'QL', verbose, timings, \
                       MAX_STEPS['ql'], NUM_TRIALS['ql'], max_episodes=QL_MAX_EPISODES, \
                       min_sub_thetas=QL_MIN_SUB_THETAS, theta=QL_THETA, epsilon_decays=QL_EPSILON_DECAYS) # NUM_TRIALS = 1000, MAX_STEPS = 30000, MAX_EPISIDES = 30000, QL_MIN_SUB_THETAS=5, QL_THETA = 0.001, QL_EPSILON_DECAYS = [0.00001]

    if args.plot:
        print('\n\n')
        if verbose:
            logger.info("----------")
        logger.info("Plotting results")
        plotting.plot_results(envs)

    print('\n\n')
    logger.info(timings)
    print('\n\n')

