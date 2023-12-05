import pyscf
from pyscf import dft, scf, cc
import numpy

def get_dc_energy(mol, hf_calc, isRestricted=True, gridLevel=5):
    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = gridLevel
    grid.build()
    dm_hf = hf_calc.make_rdm1(ao_repr=False)
    vj, vk = hf_calc.get_jk(mol, dm_hf)

    ni = dft.numint.NumInt()
    if not isRestricted:
        exh = - numpy.einsum('ij,ji', vk[0,:,:], dm_hf[0, :, :]) / 2 - numpy.einsum('ij,ji', vk[1,:,:], dm_hf[1, :, :]) / 2
        dm_hf = dm_hf[0, :, :] + dm_hf[1, :, :]
    else:
        exh = - numpy.einsum('ij,ji', vk, dm_hf) / 4
    nelec, xc_energy, vxc = ni.nr_vxc(mol=mol, grids=grid, xc_code='scan', dms=dm_hf)
    return hf_calc.e_tot + xc_energy - exh


if __name__ == '__main__':
    mol = pyscf.gto.Mole()
    mol.atom = [['H', -.7, 0, 0], ['H', .7, 0, 0]]
    # mol.basis = 'sto-3g'
    mol.basis = 'ccpvqz'
    mol.symmetry = False
    mol.charge = 0
    mol.unit = 'B'

    mol.spin = 0
    mol.build()

    # rdft = dft.UKS(mol).run()
    mf = scf.UHF(mol).run()

    print('hf, dcdft unresctricted', mf.e_tot, get_dc_energy(mol, mf, False))

    mf = scf.RHF(mol).run()
    print('hf, dcdft restricted', mf.e_tot, get_dc_energy(mol, mf, True))

    mycc = cc.CCSD(mf).run()
    print('ccsd_t energies', mycc.e_tot + mycc.ccsd_t())

    dft_res = dft.UKS(mol)
    dft_res.xc = 'scan'
    dft_res.kernel()

