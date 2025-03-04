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

def dimer_calculation(element1, element2, totalCharge, dist, functionals, basis = 'ccpvqz', Nz = 401, Nx = 401, xmax = 2.0):
    mol = pyscf.gto.Mole()
    mol.unit = 'B'
    mol.atom = [[element1, [0, 0, -dist / 2]], [element2, [0, 0, dist / 2]]]
    mol.basis = basis
    mol.symmetry = False
    mol.charge = totalCharge
    mol.build()
    nat = len(mol._atom)

    grid = np.zeros((Nz, 3))
    grid[:, 2] = np.linspace(-dist + 1, dist + 1, Nz)

    x = np.linspace(0, xmax, Nx)
    z = np.linspace(-dist, dist, Nz)
    X, Y, Z = np.meshgrid(x, [0], z)


    meshgrid_grid = np.array([X, Y, Z])
    meshgrid_grid = meshgrid_grid.reshape((3, Nx* Nz)).T

    dm_dict = dict()
    rho_dict = dict()
    rho_mesh_dict_flat = dict()
    rho_mesh_dict_dv = dict()
    rho_mesh_dict = dict()

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
        rho_mesh_dict_dv[fun] = np.reshape(rho_mesh_dict_dv[fun], (Nx, Nz))
        rho_mesh_dict[fun] = np.reshape( rho_mesh_dict_flat[fun], (Nx, Nz))

    return rho_mesh_dict, rho_mesh_dict_dv


if __name__ == '__main__':
    Nz = 401
    Nx = 401
    xmax = 2.0
    totalCharge = int(sys.argv[1])
    element1 = sys.argv[2]
    element2 = sys.argv[3]
    dist = float(sys.argv[4])
    functionals = ['lda', 'pbe', 'scan', 'b3lyp']
    basis = 'ccpvqz'
    rho, rho_dv = dimer_calculation(element1, element2, totalCharge, dist, functionals, basis, Nz, Nx, xmax)

    levels = 40
    prename = '%s_%s_%s_'%(element1, element2, basis)
    functionals.remove('cc')
    for fun in functionals:
        plt.contourf(np.linspace(-dist, dist, Nz), np.linspace(0, xmax, Nx) ,(rho[fun] - rho['cc']), levels = 40, vmin = -.03, vmax = .03, cmap='bwr')
        plt.title(fun)
        plt.colorbar()
        plt.savefig('%srhodiff_%s.pdf'%(prename, fun))
        plt.show()

    for fun in functionals:
        plt.contourf(np.linspace(-dist, dist, Nz), np.linspace(0, xmax, Nx) ,(rho_dv[fun] - rho_dv['cc']), levels = 40, vmin = -.03, vmax = .03, cmap='bwr')
        plt.title(prename + fun + ' radially integrated')
        plt.colorbar()
        plt.savefig('%sradial_%s.pdf'%(prename, fun))
        plt.show()

    eps = 1e-2
    for fun in functionals:
        plt.contourf(np.linspace(-dist, dist, Nz), np.linspace(0, xmax, Nx) , rho_dv['cc'] * (rho_dv[fun] - rho_dv['cc']) / (rho_dv['cc']**2 + eps**2), levels = 40, vmin = -.2, vmax = .2, cmap = 'bwr')
        plt.title(prename + fun + ' radially integrated relative error')
        plt.colorbar()
        plt.savefig('relative%sradial_%s.pdf'%(prename, fun))
        plt.show()