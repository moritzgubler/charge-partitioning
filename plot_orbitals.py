#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_orbs(occ, energies, center=0.0):
    occ_a = occ[0, :]
    occ_b = occ[1, :]
    energies_a = energies[0, :]
    energies_b = energies[1, :]
    scale = 0.03
    x_a = center - scale * np.ones(energies_a.shape)
    x_b = center + scale * np.ones(energies_a.shape)
    plt.scatter(x_a, energies_a, s=9000, marker="_", linewidth=2, zorder=3, c = occ_a)
    plt.scatter(x_b, energies_b, s=9000, marker="_", linewidth=2, zorder=3, c = occ_b)


x = [0, 0.3, 0.6]
# x_names = ['Na^-', '3 H_2O^-', 'Na + 3 H_2O^{2-}']
x_names = ['Na', '3 H_2O', 'Na + 3 H_2O']
files =["results/Na_ccpvdz_neutral.json",
        "results/solve_ccpvdz_neutral.json",
        "results/Na_solve_ccpvdz_neutral.json"]

for i in range(len(files)):
    with open(files[i]) as f:
        dat = json.load(f)
    occ = np.array(dat["results"]["scan"]["orb_occupancies"])
    energies = np.array(dat["results"]["scan"]["orb_energies"])
    plot_orbs(occ, energies, x[i])

plt.xticks(x, x_names)

plt.grid()
plt.show()
