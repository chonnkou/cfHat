import math
import numpy as np
import random
import hypernetx as hnx
import warnings 
warnings.simplefilter('ignore')
import copy
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from itertools import combinations
# --------below----------------generate the ER topological structure
def compute_edgenum_heterd(k_ls, d, shares, N):
	k2s = dict(zip(k_ls, shares))
	k2d = {k: k2s[k] / sum(shares) * d for k in k_ls}
	kp_dic = {k: k2d[k] / ((k / N) * math.comb(N, k)) for k in k_ls}
	k_enum = {}
	for k in k_ls:
		expected_e_num = math.comb(N, k) * kp_dic[k]
		if expected_e_num < 1000:
			actuall_e_num = np.random.binomial(n=math.comb(N, k), p=kp_dic[k])
		else:
			deviation = math.sqrt(math.comb(N, k) * kp_dic[k] * (1-kp_dic[k]) * kp_dic[k])
			actuall_e_num = np.random.normal(loc=expected_e_num, scale=deviation)
		k_enum[k] = actuall_e_num
	return k_enum

def generate_H_er(k_ls, d, shares, N):
	k_edgenum = compute_edgenum_heterd(k_ls, d, shares, N)
	hyperedgeList = []
	for k, edge_num in k_edgenum.items():
		k_edgelist = [random.sample(range(N), k) for _ in range(int(edge_num))]
		hyperedgeList.extend(k_edgelist)
	return hnx.Hypergraph(hyperedgeList)
	
# -------above-----------------generate the ER topological structure

# select AI nodes
def select_AI_nodes(H, num):
	# return the index of randomly selected AI nodes
	return list(random.sample(range(len(list(H.nodes))), num))

# return if a hyperedge is a human-ai interaction
def eifha(e, e2n, label):
	sum_n = sum([label[n] for n in e2n[e]])
	if sum_n == 0:
		return False
	else:
		return True

# eliminate the interactions that only have AI nodes
def eliminate(H, label):
	e2n = H.incidence_dict
	e2ainum = {e: sum([label[n] for n in e2n[e]]) for e in H.edges}
	e2aif = {e: num / len(e2n[e]) for e, num in e2ainum.items()}
	fullai_e = [e for e, f in e2aif.items() if f == 1.0]
	if fullai_e:
		for e in fullai_e:
			drop_ai = random.sample(e2n[e], 1)
			label[drop_ai] = 0
	return label

# aid version 
def eliminate_aid(H, label):
	e2n = H.incidence_dict
	e2ainum = {e: sum([label[n] for n in e2n[e]]) for e in H.edges}
	e2aif = {e: num / len(e2n[e]) for e, num in e2ainum.items()}
	fullai_e = [e for e, f in e2aif.items() if f == 1.0]
	if fullai_e:
		for e in fullai_e:
			n2d = {n: H.degree(n, s=1) for n in e2n[e]}
			drop_ai = min(n2d, key=n2d.get)
			label[drop_ai] = 0
	return label
	
# hd version 
def eliminate_hd(H, label):
	e2n = H.incidence_dict
	e2ainum = {e: sum([label[n] for n in e2n[e]]) for e in H.edges}
	e2aif = {e: num / len(e2n[e]) for e, num in e2ainum.items()}
	fullai_e = [e for e, f in e2aif.items() if f == 1.0]
	if fullai_e:
		for e in fullai_e:
			n2d = {n: H.degree(n, s=1) for n in e2n[e]}
			drop_ai = max(n2d, key=n2d.get)
			label[drop_ai] = 0
	return label
	
# non version - randomly place ai nodes in feasible locations
def augment(H, label, objective):
	# replenish ai nodes
	# find feasible locations and place ai nodes until there is no feasible locations
	# obtain the shortage
	shortage = int(objective - label.sum())
	e2n = H.incidence_dict
	n2e = H.dual().incidence_dict
	e2ainum = {e: sum([label[n] for n in e2n[e]]) for e in H.edges}
	e2hnum = {e: len(e2n[e]) - e2ainum[e] for e in H.edges}
	candidates = np.where(label == 0)[0].tolist()
	feasi_locations = []
	for n in candidates:
		locations = [int(e2hnum[e]) for e in n2e[n]]
		feasible = False if 1 in locations else True
		if feasible:
			feasi_locations.append(n)
	
	if len(feasi_locations) >= shortage:
		feasi_locations = random.sample(feasi_locations, shortage)
	
	label[feasi_locations] = 1
	return label
	
# aid version - place ai nodes in feasible locations as per degree
def augment_aid(H, label, objective):
	# replenish ai nodes
	# find feasible locations and place ai nodes until there is no feasible locations
	# obtain the shortage
	shortage = int(objective - label.sum())
	e2n = H.incidence_dict
	n2e = H.dual().incidence_dict
	e2ainum = {e: sum([label[n] for n in e2n[e]]) for e in H.edges}
	e2hnum = {e: len(e2n[e]) - e2ainum[e] for e in H.edges}
	candidates = np.where(label == 0)[0].tolist()
	feasi_locations = []
	for n in candidates:
		locations = [int(e2hnum[e]) for e in n2e[n]]
		feasible = False if 1 in locations else True
		if feasible:
			feasi_locations.append(n)
	
	if len(feasi_locations) >= shortage:
		n2d = {n: H.degree(n, s=1) for n in feasi_locations}
		feasi_locations = [k for k, v in sorted(n2d.items(), key=lambda item: item[1], reverse=True)]
		feasi_locations = feasi_locations[:shortage]
		
	label[feasi_locations] = 1
	return label
	
# hd version - place ai nodes in feasible locations as per degree
def augment_hd(H, label, objective):
	# replenish ai nodes
	# find feasible locations and place ai nodes until there is no feasible locations
	# obtain the shortage
	shortage = int(objective - label.sum())
	e2n = H.incidence_dict
	n2e = H.dual().incidence_dict
	e2ainum = {e: sum([label[n] for n in e2n[e]]) for e in H.edges}
	e2hnum = {e: len(e2n[e]) - e2ainum[e] for e in H.edges}
	candidates = np.where(label == 0)[0].tolist()
	feasi_locations = []
	for n in candidates:
		locations = [int(e2hnum[e]) for e in n2e[n]]
		feasible = False if 1 in locations else True
		if feasible:
			feasi_locations.append(n)
	
	if len(feasi_locations) >= shortage:
		n2d = {n: H.degree(n, s=1) for n in feasi_locations}
		feasi_locations = [k for k, v in sorted(n2d.items(), key=lambda item: item[1], reverse=False)]
		feasi_locations = feasi_locations[:shortage]
		
	label[feasi_locations] = 1
	return label
	
# non-dominant teams - heter ER
def generate_T_er(N, d, rho, k_ls, shares):
	'''
	return a human-ai team
	H_skeleton: the topological structure
	label: node label
	e2type: hyperedge label
	'''
	H_skeleton = generate_H_er(k_ls, d, shares, N)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes(H_skeleton, num)
	label[ai_nodes] = 1
	# eliminate interactions full of ai nodes
	label = eliminate(H_skeleton, label)
	# augment AI nodes to reach objective
	label = augment(H_skeleton, label, num)
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type
	

# non-dominant teams - uni ER
# determine how many k-hyperedges should appear 
def compute_edge_num(k, d, N):
	N_k = math.comb(N, k)
	k_N = k / N
	p = d / (k_N * N_k)
	expected_e_num = N_k * p
	if expected_e_num < 1000:
		actuall_e_num = np.random.binomial(n=N_k, p=p)
	else:
		deviation = math.sqrt(N_k * p * (1-p) * p)
		actuall_e_num = np.random.normal(loc=expected_e_num, scale=deviation)
	return int(actuall_e_num)

# generate a random uniform ER hypergraph
def generate_unier(k, d, N):
	edge_num = compute_edge_num(k, d, N)
	hyperedgeList = [random.sample(range(N), k) for i in range(edge_num)]
	H_uer = hnx.Hypergraph(hyperedgeList)
	return H_uer
	
# generate uniER-non
def generate_T_unier_non(N, d, rho, k):
	'''
	H_skeleton: the topological structure
	label: node label
	e2type: hyperedge label
	'''
	H_skeleton = generate_unier(k, d, N)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes(H_skeleton, num)
	label[ai_nodes] = 1
	# eliminate interactions full of ai nodes
	label = eliminate(H_skeleton, label)
	# augment AI nodes to reach objective
	label = augment(H_skeleton, label, num)
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type

# generate uniER-aid
def generate_T_unier_aid(N, d, rho, k):
	H_skeleton = generate_unier(k, d, N)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes_aid(H_skeleton, num)
	label[ai_nodes] = 1
	# eliminate interactions full of ai nodes
	label = eliminate_aid(H_skeleton, label)
	# augment AI nodes to reach objective
	label = augment_aid(H_skeleton, label, num)
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type
	
# generate uniER-hd
def generate_T_unier_hd(N, d, rho, k):
	H_skeleton = generate_unier(k, d, N)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes_hd(H_skeleton, num)
	label[ai_nodes] = 1
	# eliminate interactions full of ai nodes
	label = eliminate_hd(H_skeleton, label)
	# augment AI nodes to reach objective
	label = augment_hd(H_skeleton, label, num)
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type


# two utils
# return the ha degree of a node
def ha_deg(H, node, e2type):
	n2e = H.dual().incidence_dict
	hae_ls = [e for e in n2e[node] if e2type[e] == 1]
	return len(hae_ls)
	
# return the density of human-ai interactions of a team
def test_lambda(T):
	H_sk, label, e2type = T
	# retrive V`
	V_ = np.where(label == 0)[0].tolist()
	# <d*>
	mean_d_star = round(sum([ha_deg(H_sk, n, e2type) for n in V_]) / len(V_), 3)
	# retrive V
	V = H_sk.nodes
	# <d>
	mean_d = round(sum([H_sk.degree(n, s=1) for n in V]) / len(V), 3)
	return round(mean_d_star / mean_d, 3)
	
# --------below----------------generate the lattice topological structure
def generate_matrix(N):
	"""
	N: the number of elements; has to be 4 ** k; i.e., 256, 1024...
	"""
	k = math.log(N, 4)
	# make sure we have N nodes
	rows = 4 ** 2
	cols = 4 ** (k-2) 
	data = np.arange(N)
	matrix = data.reshape((int(rows), int(cols)))
	return matrix
	
def generate_lattice_desired(N):
	M = generate_matrix(N)
	rows, cols = M.shape
	edges = []
	for i in range(rows):
		for j in range(cols):
			cp = M[i, j]
			# declare the neighbor points
			up = M[i-1, j] if i > 0 else None
			down = M[i+1, j] if i < rows - 1 else None
			left = M[i, j-1] if j > 0 else None
			right = M[i, j+1] if j < cols - 1 else None
			# declare the quadrant points
			quadrant_1 = M[i-1, j+1] if i > 0 and j < cols - 1 else None
			quadrant_2 = M[i-1, j-1] if i > 0 and j > 0 else None
			quadrant_3 = M[i+1, j-1] if i < rows - 1 and j > 0 else None
			quadrant_4 = M[i+1, j+1] if i < rows - 1 and j < cols - 1 else None
			# declare the outer points
			upup = M[i-2, j] if i > 1 else None
			downdown = M[i+2, j] if i < rows - 2 else None
			leftleft = M[i, j-2] if j > 1 else None
			rightright = M[i, j+2] if j < cols - 2 else None
			# inner points
			if 0 < i < rows - 1 and 0 < j < cols - 1:
				edges.append([cp, up, right])
				edges.append([cp, down, left])
				# aug3
				edges.append([cp, up])
				edges.append([cp, down])
				edges.append([cp, left])
				edges.append([cp, right])
				# aug4
				# all quadrant points
				edges.append([cp, quadrant_1])
				edges.append([cp, quadrant_2])
				edges.append([cp, quadrant_3])
				edges.append([cp, quadrant_4])
				if 1 < i < rows - 2 and 1 < j < cols - 2:
					# aug5
					edges.append([cp, upup])
					edges.append([cp, downdown])
					edges.append([cp, leftleft])
					edges.append([cp, rightright])
						
			# up-edge points
			elif i == 0 and 0 < j < cols - 1:
				edges.append([cp, down, right])
				edges.append([cp, down, left])
				# aug3
				edges.append([cp, down])
				edges.append([cp, left])
				edges.append([cp, right])
				# aug4
				edges.append([cp, quadrant_3])
				edges.append([cp, quadrant_4])
				
			# down-edge points
			elif i == rows -1 and 0 < j < cols - 1:
				edges.append([cp, up, right])
				edges.append([cp, up, left])
				# aug3
				edges.append([cp, up])
				edges.append([cp, left])
				edges.append([cp, right])
				# aug4
				edges.append([cp, quadrant_1])
				edges.append([cp, quadrant_2])
				
			# left-edge points
			elif j == 0 and 0 < i < rows - 1:
				edges.append([cp, up, right])
				edges.append([cp, down, right])
				# aug3
				edges.append([cp, up])
				edges.append([cp, down])
				edges.append([cp, right])
				# aug4
				edges.append([cp, quadrant_1])
				edges.append([cp, quadrant_4])
				
			# right-edge points
			elif j == cols - 1 and 0 < i < rows - 1:
				edges.append([cp, up, left])
				edges.append([cp, down, left])
				# aug3
				edges.append([cp, up])
				edges.append([cp, down])
				edges.append([cp, left])
				# aug4
				edges.append([cp, quadrant_2])
				edges.append([cp, quadrant_3])
				
			# corner points
			else:
				neighbors = [n for n in [up, down, left, right] if n is not None]
				edges.append([cp] + neighbors)
				# aug3
				for n in neighbors:
					edges.append([cp, n])
				# aug4
				if quadrant_1:
					edges.append([cp, quadrant_1])
				if quadrant_2:
					edges.append([cp, quadrant_2])
				if quadrant_3:
					edges.append([cp, quadrant_3])
				if quadrant_4:
					edges.append([cp, quadrant_4])
				
	return hnx.Hypergraph(edges)

# --------above----------------generate the lattice topological structure
def generate_T_lattice(N, rho):
	H_skeleton = generate_lattice_desired(N)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes(H_skeleton, num)
	label[ai_nodes] = 1
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type
	
# return the average degree of ai nodes
def mean_deg_ainodes(team):
	H_sk, label, e2type = team
	ainodes = np.where(label == 1)[0].tolist()
	ainodes_degs = [H_sk.degree(n, s=1) for n in ainodes]
	return round(sum(ainodes_degs) / len(ainodes_degs), 3)
	
def mean_deg(T):
	H_sk, label, e2type = T
	degs = [H_sk.degree(n, s=1) for n in H_sk.nodes]
	return round(sum(degs) / len(degs), 3)

	
# --------below----------------generate the BA topological structure
# uniform BA
def select_nodes_by_degree(h, H):
	degrees = [H.degree(n, s=1) for n in H.nodes]
	total_deg = sum(degrees)
	probs = [d / total_deg for d in degrees]
	selected_nodes = list(np.random.choice(H.nodes, size=h, replace=False, p=probs))
	return selected_nodes

def new_edges(node, m, current_edges, k):
	H_current = hnx.Hypergraph(current_edges)
	return [select_nodes_by_degree(k-1, H_current) + [node] for _ in range(m)]
	
def generate_uba(d, N, k):
	m = math.ceil(d / k)
	# start with a fully connected core - simplicial complex - size M 
	M = max([m-1, k])
	# initial hyperedges
	hyperedges = [list(pair) for pair in combinations(range(M), k)]
	# 连续添加新节点直至节点数达到 N
	for t in range(M, N+1):
		added_edges = new_edges(t, m, hyperedges, k)
		hyperedges.extend(added_edges)
	
	return hnx.Hypergraph(hyperedges)
	
# generate uniBA-non
def generate_T_uniba_non(N, d, rho, k):
	'''
	H_skeleton: the topological structure
	label: node label
	e2type: hyperedge label
	'''
	H_skeleton = generate_uba(d, N, k)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes(H_skeleton, num)
	label[ai_nodes] = 1
	# eliminate interactions full of ai nodes
	label = eliminate(H_skeleton, label)
	# augment AI nodes to reach objective
	label = augment(H_skeleton, label, num)
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type
	
# generate uniBA-aid
def generate_T_uniba_aid(N, d, rho, k):
	H_skeleton = generate_uba(d, N, k)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes_aid(H_skeleton, num)
	label[ai_nodes] = 1
	# eliminate interactions full of ai nodes
	label = eliminate_aid(H_skeleton, label)
	# augment AI nodes to reach objective
	label = augment_aid(H_skeleton, label, num)
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type


# generate uniBA-hd
def generate_T_uniba_hd(N, d, rho, k):
	H_skeleton = generate_uba(k, d, N)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes_hd(H_skeleton, num)
	label[ai_nodes] = 1
	# eliminate interactions full of ai nodes
	label = eliminate_hd(H_skeleton, label)
	# augment AI nodes to reach objective
	label = augment_hd(H_skeleton, label, num)
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type

def select_nodes_by_degree(h, H):
	degrees = [H.degree(n, s=1) for n in H.nodes]
	total_deg = sum(degrees)
	probs = [d / total_deg for d in degrees]
	selected_nodes = list(np.random.choice(H.nodes, size=h, replace=False, p=probs))
	return selected_nodes

def new_edges(node, m, current_edges, k):
	H_current = hnx.Hypergraph(current_edges)
	return [select_nodes_by_degree(k-1, H_current) + [node] for _ in range(m)]
	
# heterogenoeus BA
def generate_hterba(d, N, k_ls, shares):
	d_1 = d * shares[0] / sum(shares)
	d_2 = d * shares[1] / sum(shares)
	# the number of 1-hyperedges
	m_1 = math.ceil(d_1 / 2)
	# # the number of 2-hyperedges
	m_2 = math.ceil(d_2 / 3)
	# compute the size of initial core
	M = max([(m_1+m_2)-1, 3])
	hyperedges = []
	# build the initial core
	initial_edges = [list(pair) for pair in combinations(range(M), 2)]
	initial_tris = [list(pair) for pair in combinations(range(M), 3)]
	hyperedges.extend(initial_edges)
	hyperedges.extend(initial_tris)
	for t in range(M, N):
		# The selection probabilities of the existing nodes are proportional to their current degrees
		added_edges = new_edges(t, m_1, hyperedges, k_ls[0])
		added_triangles = new_edges(t, m_2, hyperedges, k_ls[1])
		hyperedges.extend(added_edges)
		hyperedges.extend(added_triangles)
	return hnx.Hypergraph(hyperedges)
# --------above----------------generate the BA topological structure
	

def select_AI_nodes_aid(H, num):
	'''
	aid: AI-dominant BA team
	return top num nodes 
	'''
	n2d = {n: H.degree(n, s=1) for n in H.nodes}
	sorted_items = sorted(n2d.items(), key=lambda x: x[1], reverse=True)
	return [k for k, v in sorted_items[:num]]
	
def generate_T_ba_aid(N, d, rho, k_ls, shares):
	H_skeleton = generate_hterba(d, N, k_ls, shares)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes_aid(H_skeleton, num)
	label[ai_nodes] = 1
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type
	
def select_AI_nodes_hd(H, num):
	'''
	hd: human-dominant BA team
	return bottom num nodes 
	'''
	n2d = {n: H.degree(n, s=1) for n in H.nodes}
	sorted_items = sorted(n2d.items(), key=lambda x: x[1], reverse=False)
	return [k for k, v in sorted_items[:num]]

def generate_T_ba_hd(N, d, rho, k_ls, shares):
	H_skeleton = generate_hterba(d, N, k_ls, shares)
	label = np.zeros(len(list(H_skeleton.nodes)))
	num = int(N * rho)
	ai_nodes = select_AI_nodes_hd(H_skeleton, num)
	label[ai_nodes] = 1
	e2n = H_skeleton.incidence_dict
	e2type = {e: 1 if eifha(e, e2n, label) else 0 for e in H_skeleton.edges}
	return H_skeleton, label, e2type

	
	
	
	

	
	
	
	
	
	
	
	
	



