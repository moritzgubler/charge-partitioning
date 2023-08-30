#!/usr/bin/env python3
import sys
import pyscf
import numpy as np
import chargePartitioning.chargePartitioning as chargePartitioning
#import matplotlib.pyplot as plt
from pyscf import dft
import chargePartitioning.electronCounter as electronCounter
# import dc_dft


# sets the density of the atom centered integration grid.
# A higher number corresponds to more grid points
gridLevel = 3
# choose charge partitioning method.
# must be either 'hirshfeld' or 'voronoi'
mode = 'hirshfeld'

if len(sys.argv) != 4:
    print("""Provide three arguments, first is xyz filename of molecule, second is total charge of molecule, third is basiss""")
    quit()
xyzFilename = sys.argv[1]
totalCharge = float(sys.argv[2])

mol = pyscf.gto.Mole()
mol.atom = xyzFilename
# mol.basis = 'sto-3g'
mol.basis = sys.argv[3]
mol.symmetry = False
mol.charge = totalCharge
mol.build()

n_elec, core_elec, val_elec = electronCounter.countElectrons(mol)
core_elec = 0

# DFT pbe calculation
print("\n\nStart DFT calculation")
dft_res = dft.RKS(mol)
dft_res.xc = 'pbe'
dft_res.newton()
dft_res.kernel()
e_pbe = dft_res.e_tot
dm_dft = dft_res.make_rdm1(ao_repr=True)
# dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
charges_dft_pbe = chargePartitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
print('dft-charges', *charges_dft_pbe)
print('sum of dft charges', np.sum(charges_dft_pbe), '\n\n')
del(dft_res)
sys.stdout.flush()


# DFT scan calculation
print("\n\nStart DFT calculation")
dft_res = dft.RKS(mol)
dft_res.xc = 'scan'
dft_res.newton()
dft_res.kernel()
e_scan = dft_res.e_tot
dm_dft = dft_res.make_rdm1(ao_repr=True)
# dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
charges_dft_scan = chargePartitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
print('dft-charges', *charges_dft_scan)
print('sum of dft charges', np.sum(charges_dft_scan), '\n\n')
del(dft_res)
sys.stdout.flush()

# DFT scan calculation
print("\n\nStart DFT calculation")
dft_res = dft.RKS(mol)
dft_res.xc = 'rpbe'
dft_res.newton()
dft_res.kernel()
e_rpbe = dft_res.e_tot
dm_dft = dft_res.make_rdm1(ao_repr=True)
# dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
charges_dft_rpbe = chargePartitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
print('dft-charges', *charges_dft_rpbe)
print('sum of dft charges', np.sum(charges_dft_rpbe), '\n\n')
del(dft_res)
sys.stdout.flush()


# DFT scan calculation
print("\n\nStart DFT calculation")
dft_res = dft.RKS(mol)
dft_res.xc = 'blyp'
dft_res.newton()
dft_res.kernel()
e_blyp = dft_res.e_tot
dm_dft = dft_res.make_rdm1(ao_repr=True)
# dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
charges_dft_blyp = chargePartitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
print('dft-charges', *charges_dft_blyp)
print('sum of dft charges', np.sum(charges_dft_blyp), '\n\n')
del(dft_res)
sys.stdout.flush()


# DFT scan calculation
print("\n\nStart DFT calculation")
dft_res = dft.RKS(mol)
dft_res.xc = 'b3lyp'
dft_res.newton()
dft_res.kernel()
e_b3lyp = dft_res.e_tot
dm_dft = dft_res.make_rdm1(ao_repr=True)
# dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
charges_dft_b3lyp = chargePartitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
print('dft-charges', *charges_dft_b3lyp)
print('sum of dft charges', np.sum(charges_dft_b3lyp), '\n\n')
del(dft_res)
sys.stdout.flush()

# Hartree Fock calculation
print("Start Hartree Fock calculation")
mf = mol.RHF(max_cycle=1000).run()
dm_hf = mf.make_rdm1(ao_repr=True)
e_hf = mf.e_tot
# dm_hf = dm_hf[0, :, :] + dm_hf[1, :, :]
charges_hf = chargePartitioning.getAtomicCharges(mol, dm_hf, mode, gridLevel)
print('hf-charges', *charges_hf)
print('sum of hf charges', np.sum(charges_hf), '\n\n')
sys.stdout.flush()

# Coupled Cluster calculation
print("Start coupled cluster calculation")
mycc = mf.CCSD(frozen=core_elec)
# mycc.async_io = False
mycc.direct = True
mycc.incore_complete = True
mycc.run()
e_cc = mycc.e_tot

with open('energies.txt', mode='w') as f: 
    f.write('# pbe, scan, hf, cc, dc-dft\n')
    f.write("%f %f %f %f %f %f %f \n"%(e_pbe, e_scan, e_rpbe, e_blyp, e_b3lyp, e_hf, e_cc))

dm_cc = mycc.make_rdm1(ao_repr=True)
# dm_cc = dm_cc[0] + dm_cc[1]
sys.stdout.flush()
charges_cc = chargePartitioning.getAtomicCharges(mol, dm_cc, mode, gridLevel)
print('cc - charges', *charges_cc)
print('sum of cc charges', np.sum(charges_cc), '\n\n')
sys.stdout.flush()

with open('charges.txt', mode='w') as f:
    f.write('# pbe, scan, rpbe, blyp, b3lyp, hf, cc\n')
    for pbe, scan, rpbe, blyp, b3lyp, hf, cc in zip(charges_dft_pbe, charges_dft_scan, charges_dft_rpbe, charges_dft_blyp, charges_dft_b3lyp, charges_hf, charges_cc):
        f.write("%f %f %f %f %f %f %f\n"%(pbe, scan, rpbe, blyp, b3lyp, hf, cc))

