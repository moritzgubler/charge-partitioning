#!/usr/bin/env python3
import sys
import pyscf
import numpy as np
from pyscf import dft, cc
import matplotlib.pyplot as plt 

def get_rho(mol, dm, grid):
    ao = mol.eval_gto(eval_name='GTOval', coords=grid)
    rho = np.einsum('pi,ij,pj->p', ao, dm, ao)
    return rho

def dft_sim(mol, functional):
    dft_res = dft.UKS(mol)
    dft_res.xc = functional
    dft_res.kernel()
    dm = dft_res.make_rdm1(ao_repr = True)
    dm = dm[0, :, :] + dm[1, :, :]
    return dm

totalCharge = int(sys.argv[1])
element1 = sys.argv[2]
element2 = sys.argv[3]
dist = float(sys.argv[4])
mol = pyscf.gto.Mole()
mol.unit = 'B'
mol.atom = [[element1, [0, 0, -dist / 2]], [element2, [0, 0, dist / 2]]]
mol.basis = 'ccpvtz'
mol.symmetry = False
mol.charge = totalCharge
mol.build()
nat = len(mol._atom)

N = 801
grid = np.zeros((N, 3))
grid[:, 2] = np.linspace(-dist, dist, N)

Nx = 801
x = np.linspace(0, 6, Nx)
z = grid.copy()
X, Z = np.meshgrid(x, z)

meshgrid_grid = np.zeros((N * Nx, 3))
ii = 0
for i in range(Nx):
    for j in range(N):
        meshgrid_grid[ii, :] = np.array([X[j, i], 0, Z[j, i]])
        ii += 1


dm_dict = dict()
rho_dict = dict()
rho_mesh_dict = dict()

functionals = ['lda', 'pbe', 'scan', 'b3lyp']

mf = mol.UHF().run()
dm_hf = mf.make_rdm1(ao_repr=True)
dm_hf = dm_hf[0, :, :] + dm_hf[1, :, :]
dm_dict['hf'] = dm_hf
my_cc = cc.UCCSD(mf).run()
dm_cc = my_cc.make_rdm1(ao_repr = True)
dm_cc = dm_cc[0] + dm_cc[1]
dm_dict['cc'] = dm_cc

for fun in functionals:
    dm_dict[fun] = dft_sim(mol, fun)
functionals.append('hf')
functionals.append('cc')
for fun in functionals:
    rho_dict[fun] = get_rho(mol, dm_dict[fun], grid)
    rho_mesh_dict[fun] = get_rho(mol, dm_dict[fun], meshgrid_grid)

for fun in functionals:
    plt.plot(grid[:, 2], rho_dict[fun], label = fun)
plt.legend()
plt.show()
functionals.remove('cc')
for fun in functionals:
    plt.plot(grid[:, 2], rho_dict[fun] - rho_dict['cc'], label = fun)
plt.title('Difference to cc')
plt.legend()
plt.show()

