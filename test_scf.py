import pyscf
import numpy as np
import chargePartitioning
import matplotlib.pyplot as plt

mol = pyscf.gto.Mole()
mol.atom = 'test.xyz'
mol.unit = 'B'
#mol.basis = 'sto-3g'
mol.basis = 'ccpvtz'
mol.symmetry = False
mol.build()

mf = mol.UHF(max_cycle=1000).run()
dft = mol.UKS(max_cycle=1000)
dft.xc = 'pbe'
dft = dft.run()
mycc = mf.CCSD().run()

dm = mycc.make_rdm1(ao_repr=True)

# chargePartitioning.splitMoleculeToAtoms(mol)
nx = 1000
ny = 1
nz = 1
x = np.zeros((nx,3))
x[:,2] = np.linspace(-2,5,nx)
cp1 = chargePartitioning.partitioningWeights(x, mol, 0)
# cp2 = chargePartitioning.partitioningWeights(x, mol, 1)
# cp3 = chargePartitioning.partitioningWeights(x, mol, 2)

charges = chargePartitioning.getAtomicCharges(mol, dm)
print('charges', charges)
print('total charge,', sum(charges))

# plt.plot(x[:, 2], cp1)
# plt.show()

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
