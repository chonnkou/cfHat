This repository contains the code used to generate the simulation results in our working paper on consensus formation in human–AI teams.

### Repository Structure

- `generate_teams.py` 
  Code for **generating human–AI teams** (HATs), including interaction structures used in the simulations.

- `opinion_dynamics.py` 
  Core implementation of the **opinion dynamics models** used in the paper (including update rules and related utilities).

- `Trajectory_ClusterSize.ipynb` 
  Jupyter notebook used to reproduce the paper’s results on:
  - opinion trajectories
  - cluster-size distribution

- `tasks/` 
  A folder containing python scripts used to reproduce other experiments and analyses presented in the paper.

### Requirements

- **Python**: 3.11.9
- **Packages**:

```bash
pip install numpy matplotlib joblib hypernetx seaborn fastjsonschema
