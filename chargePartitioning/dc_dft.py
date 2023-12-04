import pyscf
from pyscf import dft
import numpy

def get_dc_energy(mol, hf_calc, isRestricted=True, gridLevel=5):
    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = gridLevel
    grid.build()
    dm_hf = hf_calc.make_rdm1(ao_repr=False)
    vj, vk = hf_calc.get_jk(mol, dm_hf)
    print(vk.shape)

    ni = dft.numint.NumInt()
    if not isRestricted:
        exh = - numpy.einsum('ij,ji->', vk[0, :, :], dm_hf[0, :, :]) / 4 - numpy.einsum('ij,ji->', vk[1, :, :], dm_hf[1, :, :]) / 4
        # dm_hf = dm_hf[0, :, :] + dm_hf[1, :, :]
        nelec, xc_energy1, vxc = ni.get_vxc(mol=mol, grids=grid, xc_code='HF, ', dms=dm_hf[0, :, :])
        nelec, xc_energy2, vxc = ni.get_vxc(mol=mol, grids=grid, xc_code='HF, ', dms=dm_hf[1, :, :])
        return hf_calc.e_tot + xc_energy1 + xc_energy2 - exh
    else:
        exh = - numpy.einsum('ij,ji->', vk, dm_hf) / 4
        nelec, xc_energy, vxc = ni.get_vxc(mol=mol, grids=grid, xc_code='HF', dms=dm_hf)
        return hf_calc.e_tot + xc_energy - exh