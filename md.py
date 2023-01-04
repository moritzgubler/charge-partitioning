#!/usr/bin/env python3
import sys
import pyscf
import numpy as np
import chargePartitioning
#import matplotlib.pyplot as plt
from pyscf import dft
import time
import pyscf.md as md


# sets the density of the atom centered integration grid.
# A higher number corresponds to more grid points
gridLevel = 5
# choose charge partitioning method.
# must be either 'hirshfeld' or 'voronoi'
mode = 'hirshfeld'

# create pyscf molecule object. Important, units must be bohr when hirshfeld partitioning is used.
if len(sys.argv) != 3:
    print("Provide two arguments, first is xyz filename of molecule, second is total charge of molecule")
    quit()
xyzFilename = sys.argv[1]
totalCharge = float(sys.argv[2])
mol = pyscf.gto.Mole()
mol.atom = xyzFilename
mol.basis = 'sto-3g'
# mol.basis = 'ccpvdz'
mol.symmetry = False
mol.charge = totalCharge
mol.build()

print("\n\nStart DFT calculation")
mf = mol.UHF().run()


grad_scan = pyscf.scf.UHF(mol).nuc_grad_method().as_scanner()

f = open('traj.xyz', mode='w')
veloc = np.zeros((26,3))
veloc[0, :] = (mol.atom_coord(1) - mol.atom_coord(0))
veloc[0, :] = veloc[0, :] / np.linalg.norm(veloc[0, :]) * 0.001
veloc[1, :] = - veloc[0, :]
integrator = md.NVE(mf, dt=40, steps=1000, trajectory_output=f ).run(veloc=veloc)
f.close()

# import sqnm

# opt = sqnm.SQNM(3*26, 10, -.1, 1e-3, 1e-2)

# for i in range(30):
#     e, g = grad_scan(mol)
#     x = mol.atom_coords()
#     xt = x[:,:]
#     gt = g[:, :]
#     xt = np.reshape(xt, 26*3)
#     gt = np.reshape(gt, 26*3)
#     xt = xt + opt.sqnm_step(xt, e, gt)
#     xt = np.reshape(xt, (26,3))

#     x[:, :] = xt[:, :]

#     atom = []

#     for i in range(26):
#         atom.append(( mol._atom[i][0] ,list(x[i, :])))

#     mol = pyscf.gto.Mole()

#     mol.atom = atom
#     mol.basis = 'sto-3g'
#     # mol.basis = 'ccpvdz'
#     mol.unit = 'B'
#     mol.symmetry = False
#     mol.charge = totalCharge    
#     mol.build()
#     for i in range(26):
#         print(mol._atom[i][0], *np.array(mol._atom[i][1]) * 0.52918)




