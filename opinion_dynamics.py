import hypernetx as hnx
import generate_teams as gene
import random
import math
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings
warnings.simplefilter('ignore')
import json
import os
from datetime import datetime
from joblib import Parallel, delayed
import hypernetx.algorithms.hypergraph_modularity as hmod
from matplotlib.lines import Line2D
import re
from scipy import stats

def generate_interval(et):
	stop, num = et
	return list(np.arange(0, stop + stop / num, stop / num))
	
	
# utils of dynamics_prototype
def opinion_distance(o, ns):
	ons = o[ns]
	return abs(np.max(ons) - np.min(ons))
	

def new_opinion(node, nodes, mapping, l, o):
	temp_ns = nodes.copy()
	temp_ns.remove(node)
	othern2trust = {n: mapping['h'] if l[n] == 0 else mapping['ai'] for n in temp_ns}
	M = sum(othern2trust.values()) + mapping['perse']
	n2weight = {n: othern2trust[n] / M for n in temp_ns}
	n2weight[node] = mapping['perse'] / M
	return sum([o[n] * n2weight[n] for n in nodes])

def update_opinion(o, ns, l, h2t, a2t):
	changes = []
	for node in ns:
		if l[node] == 0:
			# node is human node
			new_o = new_opinion(node, ns, h2t, l, o)
			changes.append(abs(new_o - o[node]))
			o[node] = new_o
		else:
			# node is AI node
			new_o = new_opinion(node, ns, a2t, l, o)
			changes.append(abs(new_o - o[node]))
			o[node] = new_o
	return o, changes
	
def trim_list(lst, max_len=1000):
	if len(lst) <= max_len:
		return lst
	return lst[-max_len:]
	
def update_opinion_h2h(o, ns):
	mean_o = np.mean(o[ns])
	changes = [abs(o[n] - mean_o) for n in ns]
	o[ns] = mean_o
	return o, changes
	
	
def check(window, N, cc):
	if len(window) == N and sum(window) < cc:
		return True
	return False
	
def update_window(window, N, changes):
	window.extend(changes)
	return trim_list(window, N)

def opinion_clusters(opinions, delta=1e-3):
	opinions = np.asarray(opinions, dtype=float)
	N = len(opinions)
	order = np.argsort(opinions)
	vals = opinions[order]

	clusters = []
	current = [order[0]]

	for i in range(1, N):
		if abs(vals[i] - vals[i-1]) <= delta:
			current.append(order[i])
		else:
			clusters.append(current)
			current = [order[i]]
	clusters.append(current)

	return clusters
	
def retrive_size_distribution(state, N):
	clusters = opinion_clusters(state)
	sizes = [len(c) / N for c in clusters]
	return sizes
	
	
def max_cluster_size(opinions, delta):
	opinions = np.asarray(opinions, dtype=float)
	N = len(opinions)
	order = np.argsort(opinions)
	vals = opinions[order]

	max_size = 1
	cur_size = 1
	for i in range(1, N):
		if abs(vals[i] - vals[i-1]) <= delta:
			cur_size += 1
		else:
			if cur_size > max_size:
				max_size = cur_size
			cur_size = 1

	if cur_size > max_size:
		max_size = cur_size
	
	return max_size / N
	
# weight mechanism
def dynamics_prototype(T, epsilon, mu, alpha, sweep):
	'''
	return relative size of the largest cluster
	'''
	H_sk, label, e2type = T
	N, cc = sweep
	e2n = H_sk.incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0.0, 1 + np.finfo(float).eps, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	# window for terminating dynamics at extremely small epsilon
	discussion_records = []
	while True:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# computer the opinion distance
		od = opinion_distance(state, nodes)
		if od >= epsilon:
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		# human-human interactions
		if label[nodes].sum() == 0:
			state, changes = update_opinion_h2h(state, nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	return max_cluster_size(state, cc)
	
	
def saving(T_type, T, mu, alpha, results, epsilon_interval, runstep, k, rho):
	H_sk, label, e2type = T
	end, num = epsilon_interval
	N = len(list(H_sk.nodes))
	save_dir = f"results_means_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"interval_length": num,
		"runstep": runstep,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}_{rho}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
	
def opinion_dynamics_means(T_type, T, epsilon_interval, mu, alpha, sweep, runstep, k, rho):
	epsilons = generate_interval(epsilon_interval)
	tasks = [(eps, i) for eps in epsilons for i in range(runstep)]
	results = Parallel(n_jobs=-1)(delayed(dynamics_prototype)(T, epsilon, mu, alpha, sweep) for epsilon, i in tasks)
	# save
	saving(T_type, T, mu, alpha, results, epsilon_interval, runstep, k, rho)
	
	
	
def saving_ana2(T_type, T, mu, alpha, results, epsilon_interval, runstep):
	H_sk, label, e2type = T
	end, num = epsilon_interval
	N = len(list(H_sk.nodes))
	save_dir = f"results_ana2_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"interval_length": num,
		"runstep": runstep,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{mu}_{alpha}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def od_ana2(T_type, T, epsilon_interval, mu, alpha, sweep, runstep):
	epsilons = generate_interval(epsilon_interval)
	tasks = [(eps, i) for eps in epsilons for i in range(runstep)]
	results = Parallel(n_jobs=59)(delayed(dynamics_prototype)(T, epsilon, mu, alpha, sweep) for epsilon, i in tasks)
	# save
	saving_ana2(T_type, T, mu, alpha, results, epsilon_interval, runstep)
	
		

def od_ana3(T, epsilon, mu, alpha, sweep):
	'''
	return states for trajectory analysis
	'''
	H_sk, label, e2type = T
	N, cc = sweep
	e2n, n2e = H_sk.incidence_dict, H_sk.dual().incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0, 1, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	# collector for state
	states = [state.copy()]
	while True:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# computer the opinion distance
		od = opinion_distance(state, nodes)
		if od >= epsilon:
			# optional - record changes only
#			states.append(state.copy())
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		# human-human interactions
		if label[nodes].sum() == 0:
			state, changes = update_opinion_h2h(state, nodes)
			states.append(state.copy())
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		states.append(state.copy())
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	return states
		
		
		
# new mechanisms
def dynamics_muepsilon(T, epsilon, mu, alpha, sweep):
	'''
	return relative size of the largest cluster
	'''
	H_sk, label, e2type = T
	N, cc = sweep
	e2n = H_sk.incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0.0, 1 + np.finfo(float).eps, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	while True:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# compute the opinion distance
		od = opinion_distance(state, nodes)
		epsilon_ = epsilon * mu if e2type[edge] == 1 else epsilon
		if od >= epsilon_:
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		# human-human interactions
		if e2type[edge] == 0:
			state, changes = update_opinion_h2h(state, nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	return max_cluster_size(state, cc)
		

def saving_ana4(T_type, T, mu, alpha, results, epsilon_interval, runstep, k, rho):
	H_sk, label, e2type = T
	end, num = epsilon_interval
	N = len(list(H_sk.nodes))
	save_dir = f"results_means_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"interval_length": num,
		"runstep": runstep,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}_{rho}_{mu}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
	
def od_ana4(T_type, T, epsilon_interval, mu, alpha, sweep, runstep, k, rho):
	epsilons = generate_interval(epsilon_interval)
	tasks = [(eps, i) for eps in epsilons for i in range(runstep)]
	results = Parallel(n_jobs=-1)(delayed(dynamics_muepsilon)(T, epsilon, mu, alpha, sweep) for epsilon, i in tasks)
	# save
	saving_ana4(T_type, T, mu, alpha, results, epsilon_interval, runstep, k, rho)
		
def retrive_cmax(o, B):
	idx = np.fromiter(B, dtype=int)
	opinions = o[idx]
	clusters = opinion_clusters(opinions, 1e-3)
	largest_cluster = max(clusters, key=len)
	return float(np.mean(largest_cluster))
		
def intervene(o, ns, B, p, eta):
#	dire = retrive_cmax(o, B)
	dire = np.mean(o[ns])
	ns_arr = np.asarray(ns, dtype=int)
	o_ns = o[ns_arr]
	steps = eta * np.abs(o_ns - dire)
	mask = (np.random.rand(ns_arr.size) < p)
	adjusts = steps * mask  # 没选中就是 0
	# changes 需要是 list，保持输出一致
	changes = adjusts.tolist()
	moves = np.where(o_ns < dire, adjusts, -adjusts)
	# 原地更新 o
	o[ns_arr] = o_ns + moves
	return o, changes
	
def intervene_cmax(o, ns, B, p, eta):
	dire = retrive_cmax(o, B)
#	dire = np.mean(o[ns])
	ns_arr = np.asarray(ns, dtype=int)
	o_ns = o[ns_arr]
	steps = eta * np.abs(o_ns - dire)
	mask = (np.random.rand(ns_arr.size) < p)
	adjusts = steps * mask  # 没选中就是 0
	# changes 需要是 list，保持输出一致
	changes = adjusts.tolist()
	moves = np.where(o_ns < dire, adjusts, -adjusts)
	# 原地更新 o
	o[ns_arr] = o_ns + moves
	return o, changes


def ai_neighbors(l, H):
	all_neighbors = set()
	for ai in np.flatnonzero(l == 1):
		all_neighbors.update(H.neighbors(ai))
	return list(all_neighbors)
		
		

def dynamics_mediator(T, epsilon, mu, alpha, sweep, p, eta, freq):
	'''
	return relative size of the largest cluster
	'''
	H_sk, label, e2type = T
	N, cc = sweep
	e2n = H_sk.incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0.0, 1 + np.finfo(float).eps, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	B = ai_neighbors(label, H_sk)
	T = 0
	while True:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# compute the opinion distance
		od = opinion_distance(state, nodes)
		if od >= epsilon:
			T += 1
			if T % freq == 0:
				# AI intervene
				state, changes = intervene(state, nodes, B, p, eta)
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		# human-human interactions
		if e2type[edge] == 0:
			state, changes = update_opinion_h2h(state, nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	return max_cluster_size(state, cc)
	
def dynamics_mediator_cmax(T, epsilon, mu, alpha, sweep, p, eta, freq):
	'''
	return relative size of the largest cluster
	'''
	H_sk, label, e2type = T
	N, cc = sweep
	e2n = H_sk.incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0.0, 1 + np.finfo(float).eps, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	B = ai_neighbors(label, H_sk)
	T = 0
	while True:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# compute the opinion distance
		od = opinion_distance(state, nodes)
		if od >= epsilon:
			T += 1
			if T % freq == 0:
				# AI intervene
				state, changes = intervene_cmax(state, nodes, B, p, eta)
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		# human-human interactions
		if e2type[edge] == 0:
			state, changes = update_opinion_h2h(state, nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	return max_cluster_size(state, cc)

		
		
def interval_extra(box):
	s, stop, num = box
	return list(np.arange(s, stop + stop / num, stop / num))	
		
def saving_ana5(T_type, results, epsilon_interval, runstep, k, freq, p, eta):
	end, num = epsilon_interval
	save_dir = f"results_means_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"interval_length": num,
		"runstep": runstep,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}_{freq}_{p}_{eta}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
		
def od_ana5(T_type, T, epsilon_interval, mu, alpha, sweep, runstep, k, freq, p, eta):
	epsilons = generate_interval(epsilon_interval)
	tasks = [(eps, i) for eps in epsilons for i in range(runstep)]
	results = Parallel(n_jobs=-1)(delayed(dynamics_mediator)(T, epsilon, mu, alpha, sweep, p, eta, freq) for epsilon, i in tasks)
	# save
	saving_ana5(T_type, results, epsilon_interval, runstep, k, freq, p, eta)
		
		
		
def saving_ana5_extra(T_type, results, epsilon_interval, runstep, k, freq, p, eta):
	s, end, num = epsilon_interval
	save_dir = f"results_means_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"interval": [s, end, num],
		"runstep": runstep,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}_{freq}_{p}_{eta}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
		
def od_ana5_extra(T_type, T, epsilon_interval, mu, alpha, sweep, runstep, k, freq, p, eta):
	epsilons = interval_extra(epsilon_interval)
	tasks = [(eps, i) for eps in epsilons for i in range(runstep)]
	results = Parallel(n_jobs=-1)(delayed(dynamics_mediator)(T, epsilon, mu, alpha, sweep, p, eta, freq) for epsilon, i in tasks)
	# save
	saving_ana5_extra(T_type, results, epsilon_interval, runstep, k, freq, p, eta)
	
def od_ana5_extra_cmax(T_type, T, epsilon_interval, mu, alpha, sweep, runstep, k, freq, p, eta):
	epsilons = interval_extra(epsilon_interval)
	tasks = [(eps, i) for eps in epsilons for i in range(runstep)]
	results = Parallel(n_jobs=-1)(delayed(dynamics_mediator_cmax)(T, epsilon, mu, alpha, sweep, p, eta, freq) for epsilon, i in tasks)
	# save
	saving_ana5_extra(T_type, results, epsilon_interval, runstep, k, freq, p, eta)

	
# extension on interval	
def saving_ana0(T_type, T, mu, alpha, results, epsilon_interval, runstep, k, rho):
	s, end, num = epsilon_interval
	save_dir = f"results_means_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"interval": [s, end, num],
		"runstep": runstep,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}_extra_base".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
		

def ana0(T_type, T, epsilon_interval, mu, alpha, sweep, runstep, k, rho):
	epsilons = interval_extra(epsilon_interval)
	tasks = [(eps, i) for eps in epsilons for i in range(runstep)]
	results = Parallel(n_jobs=-1)(delayed(dynamics_prototype)(T, epsilon, mu, alpha, sweep) for epsilon, i in tasks)
	# save
	saving_ana0(T_type, T, mu, alpha, results, epsilon_interval, runstep, k, rho)
	
# muepsilon - return states

def retrive_St(states):
	St = []
	append = St.append
	target_len = 50000
	denom = 500  # 常量提前

	for state in states:
		clusters = opinion_clusters(state)
		append(len(max(clusters, key=len)) / denom)
		if len(St) >= target_len:
			return St[:target_len]

	if St:
		last = St[-1]
		St.extend([last] * (target_len - len(St)))

	return St
	
def od_ana6(T, epsilon, mu, alpha, sweep):
	'''
	return St
	'''
	H_sk, label, e2type = T
	N, cc = sweep
	e2n, n2e = H_sk.incidence_dict, H_sk.dual().incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0, 1, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	# collector for state
	states = [state.copy()]
	while True:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# computer the opinion distance
		od = opinion_distance(state, nodes)
		epsilon_ = epsilon * mu if e2type[edge] == 1 else epsilon
		if od >= epsilon_:
			# optional - record changes only
#			states.append(state.copy())
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		# human-human interactions
		if label[nodes].sum() == 0:
			state, changes = update_opinion_h2h(state, nodes)
			states.append(state.copy())
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		states.append(state.copy())
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	return retrive_St(states)
	

	
		
def saving_ana6(T_type, k, mu, results):
	save_dir = f"results_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}_{mu}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
	
def ana6(T_type, T, epsilon, mu, alpha, sweep, runstep, k):
	results = Parallel(n_jobs=-1)(delayed(od_ana6)(T, epsilon, mu, alpha, sweep) for _ in range(runstep))
	# axis = 0 -> 平均行
	mean_St = np.mean(np.array(results), axis=0).tolist()
	
	saving_ana6(T_type, k, mu, mean_St)
	
	
def saving_ana10(T_type, k, alpha, results):
	save_dir = f"results_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}_{alpha}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
	
def ana10(T_type, T, epsilon, mu, alpha, sweep, runstep, k):
	results = Parallel(n_jobs=55)(delayed(od_ana6)(T, epsilon, mu, alpha, sweep) for _ in range(runstep))
	# axis = 0 -> 平均行
	mean_St = np.mean(np.array(results), axis=0).tolist()
	
	saving_ana10(T_type, k, alpha, mean_St)
		
		
def od_ana7(T, epsilon, mu, alpha, sweep, Tmax):
	'''
	return states for trajectory analysis
	'''
	H_sk, label, e2type = T
	N, cc = sweep
	e2n, n2e = H_sk.incidence_dict, H_sk.dual().incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0, 1, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	# collector for state
	states = [state.copy()]
	T = 1
	while T <= Tmax:
		
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# computer the opinion distance
		od = opinion_distance(state, nodes)
		epsilon_ = epsilon * mu if e2type[edge] == 1 else epsilon
		if od >= epsilon_:
			# optional - record changes only
#			states.append(state.copy())
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		T += 1
		# human-human interactions
		if label[nodes].sum() == 0:
			state, changes = update_opinion_h2h(state, nodes)
			states.append(state.copy())
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		states.append(state.copy())
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	return states
	
	
def od_ana8(T, epsilon, mu, alpha, sweep, Tmax):
	'''
	return the final state for cluster size distribution analysis
	'''
	H_sk, label, e2type = T
	Nodes_num = len(list(H_sk.nodes))
	N, cc = sweep
	e2n, n2e = H_sk.incidence_dict, H_sk.dual().incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0, 1, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	T = 0
	while T <= Tmax:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# computer the opinion distance
		od = opinion_distance(state, nodes)
		epsilon_ = epsilon * mu if e2type[edge] == 1 else epsilon
		if od >= epsilon_:
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		T += 1
		# human-human interactions
		if label[nodes].sum() == 0:
			state, changes = update_opinion_h2h(state, nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	# retrive size distribution from the final state
	return retrive_size_distribution(state, Nodes_num)
	
	
def od_ana9(T, epsilon, mu, alpha, sweep, p, eta, freq):
	'''
	return St
	'''
	H_sk, label, e2type = T
	N, cc = sweep
	e2n = H_sk.incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0.0, 1 + np.finfo(float).eps, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	B = ai_neighbors(label, H_sk)
	T = 0
	# collector for state
	states = [state.copy()]
	while True:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# compute the opinion distance
		od = opinion_distance(state, nodes)
		if od >= epsilon:
			T += 1
			if T % freq == 0:
				# AI intervene
				state, changes = intervene(state, nodes, B, p, eta)
				states.append(state.copy())
				window = update_window(window, N, changes)
				if check(window, N, cc):
					break
				continue
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		# human-human interactions
		if e2type[edge] == 0:
			state, changes = update_opinion_h2h(state, nodes)
			states.append(state.copy())
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		states.append(state.copy())
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	return retrive_St(states)
	
	
def saving_ana9(T_type, k, freq, results):
	save_dir = f"results_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}_{freq}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
	
def ana9(T_type, T, epsilon, mu, alpha, sweep, runstep, k, p, eta, freq):
	results = Parallel(n_jobs=-1)(delayed(od_ana9)(T, epsilon, mu, alpha, sweep, p, eta, freq) for _ in range(runstep))
	mean_St = np.mean(np.array(results), axis=0).tolist()
	saving_ana9(T_type, k, freq, mean_St)
	
	
def retrive_cnum(state, N):
	clusters = opinion_clusters(state)
	counter = [1 for c in clusters if len(c) / N > 0.2]
	return float(len(counter))
	
	
def od_ana11(T, epsilon, mu, alpha, sweep, Tmax=10000):
	'''
	return the final state to retrive the number of clusters 
	with relative size larger than 0.2
	'''
	H_sk, label, e2type = T
	Nodes_num = len(list(H_sk.nodes))
	N, cc = sweep
	e2n, n2e = H_sk.incidence_dict, H_sk.dual().incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0, 1, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	T = 0
	while T <= Tmax:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# computer the opinion distance
		od = opinion_distance(state, nodes)
		epsilon_ = epsilon * mu if e2type[edge] == 1 else epsilon
		if od >= epsilon_:
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		T += 1
		# human-human interactions
		if label[nodes].sum() == 0:
			state, changes = update_opinion_h2h(state, nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	# retrive number of clusters larger than 0.2
	return retrive_cnum(state, Nodes_num)
	
	
def alpha_interval(et):
	start, stop, num = et
	return list(np.arange(start, stop + stop / num, stop / num))
	
def row_modes(arr):
	return [np.unique(row, return_counts=True)[0][np.argmax(np.unique(row, return_counts=True)[1])]
			for row in arr]
			
def saving_ana11(T_type, results, k):
	save_dir = f"results_{T_type}"
	os.makedirs(save_dir, exist_ok=True)
	data = {
		"results": results,
		"timestamp": datetime.now().isoformat().split(":")[-1]
	}
	filename = f"{k}".replace(".", "_") + 't' +  data['timestamp'] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def ana11(T_type, T, epsilon, mu, alpha_info, sweep, runstep, k):
	alphas = alpha_interval(alpha_info)
	tasks = [(alpha, i) for alpha in alphas for i in range(runstep)]
	results = Parallel(n_jobs=-1)(delayed(od_ana11)(T, epsilon, mu, alpha, sweep) for alpha, i in tasks)
#	results = np.array(results)
#	results.reshape(len(alphas), runstep)
#	modes = row_modes(results)
##	means = np.mean(results, axis=1)
	saving_ana11(T_type, results, k)
	
	
def retrive_sized(state, N):
	clusters = opinion_clusters(state)
	sizes = [len(c) / N for c in clusters]
	largest = max(sizes)
	second_largest = sorted(sizes)[-2]
	return abs(largest - second_largest)
	
def od_ana12(T, epsilon, mu, alpha, sweep, Tmax=10000):
	'''
	return the final state to retrive the relative size difference
	between top1c and top2c
	'''
	H_sk, label, e2type = T
	Nodes_num = len(list(H_sk.nodes))
	N, cc = sweep
	e2n, n2e = H_sk.incidence_dict, H_sk.dual().incidence_dict
	# initial opinion - [0,1]
	state = np.random.uniform(0, 1, len(H_sk.nodes))
	# trust map of human
	h2t = {
		'perse': 1, # opinion inertia
		'h': 1,
		'ai': mu
	}
	# trust map of AI
	a2t = {
		'perse': 1, # opinion inertia
		'h': alpha,
		'ai': alpha
	}
	# window for checking convergence
	window = []
	T = 0
	while T <= Tmax:
		# randomly select a hyperedge
		edge, nodes = random.choice(list(e2n.items()))
		# computer the opinion distance
		od = opinion_distance(state, nodes)
		epsilon_ = epsilon * mu if e2type[edge] == 1 else epsilon
		if od >= epsilon_:
			changes = [0] * len(nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# update opinion
		T += 1
		# human-human interactions
		if label[nodes].sum() == 0:
			state, changes = update_opinion_h2h(state, nodes)
			window = update_window(window, N, changes)
			if check(window, N, cc):
				break
			continue
		# human-AI interactions
		state, changes = update_opinion(state, nodes, label, h2t, a2t)
		window = update_window(window, N, changes)
		if check(window, N, cc):
			break
		
	# retrive number of clusters larger than 0.2
	return retrive_sized(state, Nodes_num)
		
		
def ana12(T_type, T, epsilon, mu, alpha_info, sweep, runstep, k):
	alphas = alpha_interval(alpha_info)
	tasks = [(alpha, i) for alpha in alphas for i in range(runstep)]
	results = Parallel(n_jobs=-1)(delayed(od_ana12)(T, epsilon, mu, alpha, sweep) for alpha, i in tasks)
	saving_ana11(T_type, results, k)
		