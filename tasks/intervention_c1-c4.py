import generate_teams as gene
import opinion_dynamics as od

def main():
	T_type = 'mediators_extra'
	# fixed hypergraph paras
	ks = [3, 4, 5]
	N = 500
	d = 10
	rho = 0.2
	# fixed dynamics paras
	mu = 1
	alpha = 1
	epsilon_interval = (0.1, 0.4, 100)
	runstep = 500
	sweep = (1000, 1e-3)
	# native parameters
	freqs = [300 600 1200 2400]
	ps = [0.01, 0.1, 0.2, 0.4]
	etas = [0.05, 0.1, 0.2, 0.4]

	for k in ks:
		T = gene.generate_T_unier_non(N, d, rho, k)
		for i in range(4):
			od.od_ana5_extra(T_type, T, epsilon_interval, mu, alpha, sweep, runstep, k, freqs[i], ps[i], etas[i])
	
	

if __name__ == "__main__":
	main()



