#!/usr/bin/env python3
import sys
import pyscf
import numpy as np
import chargePartitioning.Partitioning as Partitioning
#import matplotlib.pyplot as plt
from pyscf import dft
import chargePartitioning.electronCounter as electronCounter
# import dc_dft
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# sets the density of the atom centered integration grid.
# A higher number corresponds to more grid points
gridLevel = 5
# choose charge partitioning method.
# must be either 'hirshfeld' or 'voronoi'
mode = 'hirshfeld'

restricted = False

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
mol.spin = 0
mol.build()

n_elec, core_elec, val_elec = electronCounter.countElectrons(mol)
core_elec = 0

def DFT_charges(mol, functional, restricted: bool, gridLevel = 5, mode = 'hirshfeld'):
    if restricted:
        dft_res = dft.RKS(mol)
    else:
        dft_res = dft.UKS(mol)
    dft_res.xc = functional
    dft_res.newton()
    dft_res.kernel()
    e_pot = dft_res.e_tot
    dm_dft = dft_res.make_rdm1(ao_repr=True)
    # _, e_elec = dft_res.energy_elec(dm_dft)
    if not restricted:
        dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
    dft_temp = mol.RHF()
    _, e_elec = dft_temp.energy_elec(dm_dft)
    charges = Partitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel)
    sys.stdout.flush()
    return e_pot, charges, e_elec, dm_dft

results = dict()
settings = dict()
functionals = ['lda', 'pbe', 'pbe0', 'scan', 'rscan', 'r2scan', 'blyp', 'b3lyp']

settings['basisset'] = mol.basis
settings['restricted'] = restricted
settings['total_charge'] = mol.charge
settings['spin'] = mol.spin
settings['gridlevel'] = gridLevel

for functional in functionals:
    print("Start %s calculation"%functional)
    results[functional] = dict()
    e_tot, charges, e_coul, dm_dft = DFT_charges(mol, functional, restricted, gridLevel)
    results[functional]['e_tot'] = e_tot
    results[functional]['charges'] = charges
    results[functional]['e_coulomb'] = e_coul
    results[functional]['density_matrix'] = dm_dft
    print('calculation of function %s done'%functional)
    print('coulomb energy', e_coul)
    # print('dft-charges', *charges)
    print('sum of dft charges', np.sum(charges), '\n')
    sys.stdout.flush()

# Hartree Fock calculation
print("Start Hartree Fock calculation")
if restricted:
    mf = mol.RHF(max_cycle=1000).run()
else:
    mf = mol.UHF(max_cycle=1000).run()
dm_hf = mf.make_rdm1(ao_repr=True)
e_hf = mf.e_tot
if not restricted:
    dm_hf = dm_hf[0, :, :] + dm_hf[1, :, :]
charges_hf = Partitioning.getAtomicCharges(mol, dm_hf, mode, gridLevel)
_, e_coul = mf.energy_elec(dm_hf)
print('coulomb energy', e_coul)
# print('hf-charges', *charges_hf)
print('sum of hf charges', np.sum(charges_hf), '\n\n')
sys.stdout.flush()

results['hf'] = dict()
results['hf']['charges'] = charges_hf
results['hf']['e_tot'] = e_hf
results['hf']['e_coulomb'] = e_coul
results['hf']['density_matrix'] = dm_hf
# e_dcdft = dc_dft.get_dc_energy(mol, mf, isRestricted=True, gridLevel=gridLevel)
# print('e_dcdft', e_dcdft)
# sys.stdout.flush()

# Coupled Cluster calculation
print("Start coupled cluster calculation")
mycc = mf.CCSD(frozen=core_elec)
# mycc.async_io = False
mycc.direct = True
mycc.incore_complete = True
mycc.run()
e_cc = mycc.e_tot

dm_cc = mycc.make_rdm1(ao_repr=True)
if not restricted:
    dm_cc = dm_cc[0] + dm_cc[1]
sys.stdout.flush()
charges_cc = Partitioning.getAtomicCharges(mol, dm_cc, mode, gridLevel)
_, e_coul = mf.energy_elec(dm_cc)
print('coulomb energy', e_coul)
# print('cc - charges', *charges_cc)
print('sum of cc charges', np.sum(charges_cc), '\n\n')
sys.stdout.flush()

results['cc'] = dict()
results['cc']['charges'] = charges_cc
results['cc']['e_tot'] = e_cc
results['cc']['e_coulomb'] = e_coul
results['cc']['density_matrix'] = dm_cc
results['cc']['ediff'] = 0.0
results['cc']['charge_ediff'] = 0.0


_, ediff = mf.energy_elec(dm_hf - dm_cc)
results['hf']['charge_ediff'] = ediff
results['hf']['ediff'] = results['hf']['e_coulomb'] - results['cc']['e_coulomb']
print('hf ediff', ediff, results['hf']['ediff'])

for functional in functionals:
    dft_res = mol.RHF()
    # print('%s ecoul'%functional, dft_res.energy_elec(results[functional]['density_matrix']))
    results[functional]['ediff'] = results[functional]['e_coulomb'] - results['cc']['e_coulomb']
    _, temp = dft_res.energy_elec(results[functional]['density_matrix'] - results['cc']['density_matrix'])
    print('adding result')
    results[functional]['charge_ediff'] = temp
    print('%s ecdiff'%functional, results[functional]['charge_ediff'], results[functional]['ediff'])

summary = dict()
summary['settings'] = settings
summary['results'] = results
print('\n')
with open("result.json", "w") as f:
    json.dump(summary, f, indent=4, sort_keys=True, cls = NumpyArrayEncoder)