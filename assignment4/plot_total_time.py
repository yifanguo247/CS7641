import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def read_files(alg_name, problem_name):
	times = {} # discount_factor_to_time
	xs = np.linspace(0, 0.9, 10)
	files = glob.glob('output/{}/{}_[0-9]*.[0-9].csv'.format(alg_name, problem_name)) # training files
	for i, file  in enumerate(sorted(files)):
		df = pd.read_csv(file)
		converged = len(df[df['converged']]) > 0
		times[xs[i]] = (np.sum(df['time']), converged)
	return times

def read_evaluation_files(alg_name, problem_name):
	times = {} # discount_factor_to_time
	xs = np.linspace(0, 0.9, 10)
	files = glob.glob('output/{}/{}_[0-9]*.[0-9]_optimal.csv'.format(alg_name, problem_name)) # evaluation files
	for i, file  in enumerate(sorted(files)):
		df = pd.read_csv(file)
		trial_num, steps_to_reach_goal, steps_to_reach_goal_mean, avg_step_reward, mean_avg_step_reward, median_avg_step_reward, std_avg_step_reward, max_avg_step_reward, min_avg_step_reward  = df.iloc[-1] # final trial #, average step reward in final trail, average step reward averaged over all trials so far, ...
		times[xs[i]] = (steps_to_reach_goal_mean, mean_avg_step_reward)
	return times

def plot_evaluation_results(results):
	if not os.path.exists('output/special_images/'): os.mkdir('output/special_images/')

	for alg_prob in results: # PI on cliff, PI on large_frozen_lake
		title = alg_prob
		alg, prob = alg_prob.split(' on ')
		plt.figure()
		plt.title(title)
		plt.xlabel("discount_factor")
		# plt.ylabel("reward averaged over steps per trial and over trials")
		plt.ylabel("Num Steps to Reach Goal")
		plt.grid()
		plt.tight_layout()
		for discount_factor in results[alg_prob]:
			steps_to_reach_goal_mean, mean_avg_step_reward = results[alg_prob][discount_factor]
			print('steps_to_reach_goal_mean: {}, mean_avg_step_reward: {}'.format(steps_to_reach_goal_mean, mean_avg_step_reward))
			if steps_to_reach_goal_mean != 10000:
				plt.plot(discount_factor, steps_to_reach_goal_mean, '*')
		file_name = 'output/special_images/{}_evaluation.png'.format(alg_prob)
		plt.savefig(file_name, format='png')
		plt.close()


def plot_results(results):
	if not os.path.exists('output/special_images/'): os.mkdir('output/special_images/')

	for alg_prob in results: # PI on cliff, PI on large_frozen_lake
		title = alg_prob
		alg, prob = alg_prob.split(' on ')
		plt.figure()
		plt.title(title)
		plt.xlabel("discount_factor")
		plt.ylabel("time (s)")
		plt.grid()
		plt.tight_layout()
		for discount_factor in results[alg_prob]:
			time, converged = results[alg_prob][discount_factor]
			print('discount_factor: {}, time: {}'.format(discount_factor, time))
			# color = 'blue' if converged else 'red'
			if converged:
				plt.plot(discount_factor, time, '*')
		file_name = 'output/special_images/{}.png'.format(alg_prob)
		plt.savefig(file_name, format='png')
		plt.close()

if __name__ == '__main__':
	results = {}
	evaluation_results = {}
	alg_names = ['PI', 'VI']
	prob_names = ['cliff_walking', 'large_frozen_lake', 'frozen_lake']
	for alg_name in alg_names:
		for prob_name in prob_names:
			results['{} on {}'.format(alg_name, prob_name)] = read_files(alg_name, prob_name)
			evaluation_results['{} on {}'.format(alg_name, prob_name)] = read_evaluation_files(alg_name, prob_name)

	print('results: {}'.format(results))
	# plot_results(results)
	plot_evaluation_results(evaluation_results)
