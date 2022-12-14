#!/usr/bin/env python3
import sys
import pyscf
import numpy as np
import chargePartitioning
import matplotlib.pyplot as plt
from pyscf import dft


# sets the density of the atom centered integration grid.
# A higher number corresponds to more grid points
gridLevel = 3
# choose charge partitioning method.
# must be either 'hirshfeld' or 'voronoi'
mode = 'hirshfeld'

if len(sys.argv) != 3:
    print("Provide two arguments, first is xyz filename of molecule, second is total charge of molecule")
    quit()
xyzFilename = sys.argv[1]
totalCharge = float(sys.argv[2])

mol = pyscf.gto.Mole()
mol.atom = xyzFilename
# mol.basis = 'sto-3g'
mol.basis = 'ccpvdz'
mol.symmetry = False
mol.charge = totalCharge
mol.unit = 'B'
mol.build()

# DFT calculation
print("\n\nStart DFT calculation")
dft_res = dft.UKS(mol)
dft_res.xc = 'pbe'
dft_res.newton()
dft_res.kernel()
dm_dft = dft_res.make_rdm1(ao_repr=True)
dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
charges_dft = chargePartitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
print('dft-charges', charges_dft)
print('sum of dft charges', np.sum(charges_dft), '\n\n')

# Hartree Fock calculation
print("Start Hartree Fock calculation")
mf = mol.UHF(max_cycle=1000).run()
dm_hf = mf.make_rdm1(ao_repr=True)
dm_hf = dm_hf[0, :, :] + dm_hf[1, :, :]
charges_hf = chargePartitioning.getAtomicCharges(mol, dm_hf, mode, gridLevel)
print('hf-charges', charges_hf)
print('sum of hf charges', np.sum(charges_hf), '\n\n')


# Coupled Cluster calculation
print("Start coupled cluster calculation")
mycc = mf.CCSD().run()
dm_cc = mycc.make_rdm1(ao_repr=True)
dm_cc = dm_cc[0] + dm_cc[1]
charges_cc = chargePartitioning.getAtomicCharges(mol, dm_cc, mode, gridLevel)
print('cc - charges', charges_cc)
print('sum of cc charges', np.sum(charges_cc), '\n\n')


quit()

mf = mol.UHF().run()
mycc = mf.CCSD().run()

dm1 = mycc.make_rdm1(ao_repr=True)


grids = pyscf.dft.gen_grid.Grids(mol)
grids.level = 4
grids.build()

weights = grids.weights

print('c', grids.coords)


plt.scatter(grids.coords[:,1], grids.coords[:,2], s=1)
plt.show()
# for g in grids.atom_grid:
#     print('g', g)

print(weights)

nx = 100
ny = 1
nz = 1
coords = np.zeros((nx,3))
coords[:,2] = np.linspace(-2,5,nx)

ngrids = 1
blksize = min(8000, ngrids)
# = np.empty(ngrids)


ao = mol.eval_gto(eval_name='GTOval', coords=coords)
rhoA = np.einsum('pi,ij,pj->p', ao, dm1[0], ao)
rhoB = np.einsum('pi,ij,pj->p', ao, dm1[1], ao)

# print('dm', dm1[0])
# print('dm', dm1[1])
plt.plot(coords[:,2], rhoA)
plt.plot(coords[:,2], rhoB)
plt.show()
