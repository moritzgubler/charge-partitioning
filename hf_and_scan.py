from pyscf import gto, dft
import pyscf
import numpy
import electronCounter
import dc_dft
mol = pyscf.gto.Mole()
mol.atom = "small.xyz"
mol.charge = 1
mol.basis = 'augccpvtz'
mol.build()

n_elec, core_elec, val_elec = electronCounter.countElectrons(mol)

grids = dft.gen_grid.Grids(mol)

mf = mol.RHF().run()

mycc = mf.CCSD(frozen = core_elec).run()

dc_dft_energy = dc_dft.get_dc_energy(mol, mf)

print("CCSD energy: ", mycc.e_tot)
print("dcDFT energy:", dc_dft_energy)
print('diff', mycc.e_tot - dc_dft_energy)
