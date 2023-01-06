import pyscf
from pyscf import dft
import numpy

def get_dc_energy(mol, hf_calc, isRestricted=True, gridLevel=5):
    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = gridLevel
    grid.build()
    dm_hf = hf_calc.make_rdm1(ao_repr=False)
    vj, vk = hf_calc.get_jk(mol, dm_hf)

    exh = - numpy.einsum('ij,ji->', vk, dm_hf) / 4

    if not isRestricted:
        dm_hf = dm_hf[0] + dm_hf[1]


    ni = dft.numint.NumInt()
    nelec, xc_energy, vxc = ni.get_vxc(mol=mol, grids=grid, xc_code='SCAN,SCAN', dms=dm_hf)
    return hf_calc.e_tot + xc_energy - exh
