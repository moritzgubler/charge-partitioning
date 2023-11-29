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
mol.basis = 'ccpvqz'
mol.symmetry = False
mol.charge = totalCharge
mol.build()
nat = len(mol._atom)

N = 401
grid = np.zeros((N, 3))
grid[:, 2] = np.linspace(-dist, dist, N)

Nx = 401
xmax = 2.0
x = np.linspace(0, xmax, Nx)
z = np.linspace(-dist, dist, N)
X, Y, Z = np.meshgrid(x, [0], z)


meshgrid_grid = np.array([X, Y, Z])
meshgrid_grid = meshgrid_grid.reshape((3, Nx* N)).T

dm_dict = dict()
rho_dict = dict()
rho_mesh_dict_flat = dict()
rho_mesh_dict_dv = dict()
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
    rho_mesh_dict_flat[fun] = get_rho(mol, dm_dict[fun], meshgrid_grid)
    rho_mesh_dict_dv[fun] = rho_mesh_dict_flat[fun] * meshgrid_grid[:, 0] * 2 * np.pi
    rho_mesh_dict_dv[fun] = np.reshape(rho_mesh_dict_dv[fun], (Nx, N))
    rho_mesh_dict[fun] = np.reshape( rho_mesh_dict_flat[fun], (Nx, N))

for fun in functionals:
    plt.plot(grid[:, 2], rho_dict[fun], label = fun)
plt.legend()
plt.savefig('rholine.pdf')
# plt.show()
# functionals.remove('cc')
for fun in functionals:
    plt.plot(grid[:, 2], rho_dict[fun] - rho_dict['cc'], label = fun)
plt.title('Difference to cc')
plt.legend()
plt.savefig('rholine_error.pdf')
# plt.show()
levels = 40
prename = '%s_%s_%s_'%(sys.argv[2], sys.argv[3], mol.basis)
for fun in functionals:
    plt.contourf(np.linspace(-dist, dist, N), np.linspace(0, xmax, Nx) ,(rho_mesh_dict[fun] - rho_mesh_dict['cc']), levels = 40, vmin = -.03, vmax = .03, cmap='bwr')
    plt.title(fun)
    plt.colorbar()
    plt.savefig('%srhodiff_%s.pdf'%(prename, fun))
    plt.show()

for fun in functionals:
    plt.contourf(np.linspace(-dist, dist, N), np.linspace(0, xmax, Nx) ,(rho_mesh_dict_dv[fun] - rho_mesh_dict_dv['cc']), levels = 40, vmin = -.03, vmax = .03, cmap='bwr')
    plt.title(prename + fun + ' radially integrated')
    plt.colorbar()
    plt.savefig('%sradial_%s.pdf'%(prename, fun))
    plt.show()

    eps = 1e-4


for fun in functionals:
    plt.contourf(np.linspace(-dist, dist, N), np.linspace(0, xmax, Nx) , rho_mesh_dict['cc'] * (rho_mesh_dict_dv[fun] - rho_mesh_dict_dv['cc']) / (rho_mesh_dict_dv['cc']**2 + eps**2), levels = 40, vmin = -.2, vmax = .2, cmap = 'bwr')
    plt.title(prename + fun + ' radially integrated relative error')
    plt.colorbar()
    plt.savefig('relative%sradial_%s.pdf'%(prename, fun))
    plt.show()