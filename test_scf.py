import pyscf
import numpy as np
import chargePartitioning
import matplotlib.pyplot as plt
from pyscf import gto, dft

mol = pyscf.gto.Mole()
mol.atom = 'min.xyz'
# mol.basis = 'sto-3g'
mol.basis = 'ccpvdz'
mol.symmetry = False
mol.charge = 1
mol.build()

dft_res = dft.UKS(mol)
dft_res.xc = 'lda'
dft_res.newton()
dft_res.kernel()

dm_dft = dft_res.make_rdm1(ao_repr=True)
# print(np.shape(dm_dft))
dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]

charges_dft = chargePartitioning.getAtomicCharges(mol, dm_dft)
print('dft-charges', charges_dft)
print('sum of dft charges', np.sum(charges_dft))

quit()


mf = mol.UHF(max_cycle=1000).run()

dm_hf = mf.make_rdm1(ao_repr=True)
dm_hf = dm_hf[0, :, :] + dm_hf[1, :, :]


charges_hf = chargePartitioning.getAtomicCharges(mol, dm_hf)
print('hf-charges', charges_hf)
print('sum of hf charges', np.sum(charges_hf))

mycc = mf.CCSD().run()
dm_cc = mycc.make_rdm1(ao_repr=True)
dm_cc = dm_cc[0] + dm_cc[1]

charges_cc = chargePartitioning.getAtomicCharges(mol, dm_cc)
print('cc - charges', charges_cc)
print('sum of cc charges', np.sum(charges_cc))


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
