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
nat = len(mol._atom)
mf = mol.RHF(max_cycle=5000).run()


grad_scan = mol.RHF( max_cycle=5000).nuc_grad_method().as_scanner()


# f = open('traj.xyz', mode='w')
# veloc = np.random.random((nat,3)) - 0.5
# veloc = veloc * 0.002
# veloc[0, :] = (mol.atom_coord(1) - mol.atom_coord(0))
# veloc[0, :] = -veloc[0, :] / np.linalg.norm(veloc[0, :]) * 0.0013
# veloc[1, :] = - veloc[0, :]
# integrator = md.NVE(mf, dt=50, steps=4000, trajectory_output=f ).run(veloc)
# f.close()

import sqnm

print('nat', len(mol._atom))
nat = len(mol._atom)

opt = sqnm.SQNM(3*nat, 10, -.2, 1e-3, 1e-2)
for i in range(60):
    e, g = grad_scan(mol)
    print(e, np.linalg.norm(g))
    print(nat, file=open('opt.xyz', mode='a'))
    print(e, np.linalg.norm(g), file=open('opt.xyz', mode='a'))
    for i in range(nat):
        print(mol._atom[i][0], *np.array(mol._atom[i][1]) * 0.52918, file=open('opt.xyz', mode='a'))
    x = mol.atom_coords()
    x = np.reshape(x, 3 * nat)
    g = np.reshape(g, 3 * nat)
    x = x + opt.sqnm_step(x, e, g)
    x = np.reshape(x, (nat, 3))
    g = np.reshape(g, (nat, 3))

    atom = []
    for i in range(nat):
        atom.append((mol._atom[i][0] ,list(x[i, :])))

    mol = pyscf.gto.Mole()

    mol.atom = atom
    mol.basis = 'sto-3g'
    # mol.basis = 'ccpvdz'
    mol.unit = 'B'
    mol.symmetry = False
    mol.charge = totalCharge    
    mol.build()

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




