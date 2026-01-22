import generate_teams as gene
import opinion_dynamics as od
from joblib import Parallel, delayed

def main():
	T_type = 'uniER-non'
	# fixed hypergraph paras
	N = 500
	d = 10
	# fixed dynamics paras
	epsilon_interval = (0.6, 100)
	mu = 1.0
	alpha = 1.0
	sweep = (1000, 1e-3)
	runstep = 500
	# observable parameters
	ks = [2, 3, 4, 5]
	rhos = [0.01, 0.05, 0.25, 0.5]
	
	for k in ks:
		for rho in rhos:
			T = gene.generate_T_unier_non(N, d, rho, k)
			od.opinion_dynamics_means(T_type, T, epsilon_interval, mu, alpha, sweep, runstep, k, rho)


if __name__ == "__main__":
    main()



