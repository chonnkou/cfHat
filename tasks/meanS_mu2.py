import generate_teams as gene
import opinion_dynamics as od
from joblib import Parallel, delayed

def main():
	T_type = 'muepsilon'
	# fixed hypergraph paras
	N = 500
	d = 10
	rho = 0.2
	# fixed dynamics paras
	epsilon_interval = (0.6, 100)
	mu = 2.0
	alpha = 1.0
	sweep = (1000, 1e-3)
	runstep = 500
	# observable parameters
	ks = [2, 3, 4, 5]
	Ts = [gene.generate_T_unier_non(N, d, rho, k) for k in ks]
	for i in range(4):
		od.od_ana4(T_type, Ts[i], epsilon_interval, mu, alpha, sweep, runstep, ks[i], rho)
	


if __name__ == "__main__":
    main()



