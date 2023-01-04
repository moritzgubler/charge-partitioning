from pyscf import gto, dft
import pyscf
import numpy
import electronCounter
mol = pyscf.gto.Mole()
mol.atom = "small.xyz"
mol.charge = 1
mol.basis = 'ccpvqz'
mol.build()

n_elec, core_elec, val_elec = electronCounter.countElectrons(mol)

grids = dft.gen_grid.Grids(mol)

mf = mol.RHF().run()

dm_hf = mf.make_rdm1(ao_repr=False)
# dm_hf = (dm_hf[0, :, :], dm_hf[1, :, :])

vj, vk = mf.get_jk(mol, dm_hf)

# unrestricted calculation
# exh = -(numpy.einsum('ij,ji->', vk[0], dm_hf[0]) +
#              numpy.einsum('ij,ji->', vk[1], dm_hf[1])) / 4
# restricted calculation
exh = - numpy.einsum('ij,ji->', vk, dm_hf) / 4

# et = mycc.ccsd_t()
# print('ccsd_t', mycc.e_tot + et)

ni = dft.numint.NumInt()
nelec, xc_energy, vxc = ni.get_vxc(mol=mol, grids=grids, xc_code='SCAN,SCAN', dms=dm_hf)

nelec, scan_x, vxc = ni.get_vxc(mol=mol, grids=grids, xc_code='SCAN,', dms=dm_hf)
nelec, scan_c, vxc = ni.get_vxc(mol=mol, grids=grids, xc_code=',SCAN', dms=dm_hf)



print("etot, scan-xc, hf-x, delta", mf.e_tot + xc_energy - exh, xc_energy, exh, xc_energy - exh, scan_x, scan_c)
dc_dft_energy = mf.e_tot + xc_energy - exh
# quit()
mycc = mf.CCSD(frozen = core_elec).run()


print("CCSD energy: ", mycc.e_tot)
print("dcDFT energy:", dc_dft_energy)
print('diff', mycc.e_tot - dc_dft_energy)
