#!/usr/bin/env python3
import sys
import pyscf
import numpy as np
import chargePartitioning
#import matplotlib.pyplot as plt
from pyscf import dft


# sets the density of the atom centered integration grid.
# A higher number corresponds to more grid points
gridLevel = 5
# choose charge partitioning method.
# must be either 'hirshfeld' or 'voronoi'
mode = 'hirshfeld'

if len(sys.argv) != 4:
    print("""Provide three arguments, first is xyz filename of molecule, second is total charge of molecule, third is
    frozen orbital level (should be zero or the total number of core electrons)""")
    quit()
xyzFilename = sys.argv[1]
totalCharge = float(sys.argv[2])
core_count = int(sys.argv[3])

mol = pyscf.gto.Mole()
mol.atom = xyzFilename
# mol.basis = 'sto-3g'
mol.basis = 'ccpvdz'
mol.symmetry = False
mol.charge = totalCharge
mol.build()

# DFT pbe calculation
print("\n\nStart DFT calculation")
dft_res = dft.UKS(mol)
dft_res.xc = 'pbe'
dft_res.newton()
dft_res.kernel()
dm_dft = dft_res.make_rdm1(ao_repr=True)
dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
charges_dft_pbe = chargePartitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
print('dft-charges', *charges_dft_pbe)
print('sum of dft charges', np.sum(charges_dft_pbe), '\n\n')
del(dft_res)
with open('pbe-'+ mol.basis +'.npy', 'wb') as f:
    np.save(f, dm_dft)

# DFT scan calculation
print("\n\nStart DFT calculation")
dft_res = dft.UKS(mol)
dft_res.xc = 'scan'
dft_res.newton()
dft_res.kernel()
dm_dft = dft_res.make_rdm1(ao_repr=True)
dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
charges_dft_scan = chargePartitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
print('dft-charges', *charges_dft_scan)
print('sum of dft charges', np.sum(charges_dft_scan), '\n\n')
del(dft_res)
with open('scan-'+ mol.basis +'.npy', 'wb') as f:
    np.save(f, dm_dft)

# Hartree Fock calculation
print("Start Hartree Fock calculation")
mf = mol.UHF(max_cycle=1000).run()
dm_hf = mf.make_rdm1(ao_repr=True)
dm_hf = dm_hf[0, :, :] + dm_hf[1, :, :]
charges_hf = chargePartitioning.getAtomicCharges(mol, dm_hf, mode, gridLevel)
print('hf-charges', *charges_hf)
print('sum of hf charges', np.sum(charges_hf), '\n\n')
with open('hf-'+ mol.basis +'.npy', 'wb') as f:
    np.save(f, dm_dft)


# Coupled Cluster calculation
print("Start coupled cluster calculation")
mycc = mf.CCSD(frozen=core_count).run()
dm_cc = mycc.make_rdm1(ao_repr=True)
dm_cc = dm_cc[0] + dm_cc[1]
charges_cc = chargePartitioning.getAtomicCharges(mol, dm_cc, mode, gridLevel)
print('cc - charges', *charges_cc)
print('sum of cc charges', np.sum(charges_cc), '\n\n')
with open('cc-'+ mol.basis +'.npy', 'wb') as f:
    np.save(f, dm_dft)


f = open('charges.txt', mode='w')
f.write('# pbe, scan, hf, cc\n')
for pbe, scan, hf, cc in zip(charges_dft_pbe, charges_dft_scan, charges_hf, charges_cc):
    f.write("%f  %f %f %f \n"%(pbe, scan, hf, cc))
f.close()

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
