import generate_teams as gene
import opinion_dynamics as od

def main():
	T_type = 'mueps_st'
	# fixed hypergraph paras
	ks = [2, 3, 4, 5]
	N = 500
	d = 10
	rho = 0.2
	# fixed dynamics paras
	alpha = 1
	# near the transition
	epsilons = [0.27, 0.294, 0.228, 0.312]
	runstep = 100
	sweep = (1000, 1e-3)
	# observable parameters
	mus = [0.5, 1.0, 1.5, 2.0]
	# teams
	Ts = [gene.generate_T_unier_non(N, d, rho, k) for k in ks]
	for i, T in enumerate(Ts):
		for mu in mus:
			od.ana6(T_type, T, epsilons[i], mu, alpha, sweep, runstep, ks[i])
			

if __name__ == "__main__":
	main()



