
'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

import pyscf
import numpy as np
import matplotlib.pyplot as plt

mol = pyscf.gto.Mole()
mol.atom = """
    H   0. 0. 1.5
    H   0. 0. -1.5
    O 0 0 0
"""
mol.unit = 'B'
#mol.basis = 'sto-3g'
mol.basis = 'ccpvtz'
mol.symmetry = False
mol.build()


mf = mol.UHF().run()
mycc = mf.CCSD().run()

dm1 = mycc.make_rdm1(ao_repr=True)


nx = 100
ny = 1
nz = 1
coords = np.zeros((nx,3))
coords[:,2] = np.linspace(-2,5,nx)

ngrids = 1
blksize = min(8000, ngrids)
rho = np.empty(ngrids)


ao = mol.eval_gto(eval_name='GTOval', coords=coords)
print(ao.shape)
rhoA = np.einsum('pi,ij,pj->p', ao, dm1[0], ao)
rhoB = np.einsum('pi,ij,pj->p', ao, dm1[1], ao)

print('dm', dm1[0])
print('dm', dm1[1])
plt.plot(coords[:,2], rhoA)
plt.plot(coords[:,2], rhoB)
plt.show()
