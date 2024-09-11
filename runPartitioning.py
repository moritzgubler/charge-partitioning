#!/usr/bin/env python3
import sys
import pyscf
import numpy as np
import chargePartitioning.Partitioning as Partitioning
#import matplotlib.pyplot as plt
from pyscf import dft
from pyscf import cc
import chargePartitioning.electronCounter as electronCounter
# import dc_dft
import json
from json import JSONEncoder
import os
from ase.io import read, write
from ase.atoms import Atoms
import chargePartitioning.dc_dft as dc_dft

def getElectricEnergy(scf, mol, dm):
    # get coulomb operator in matrix form
    vj = scf.get_j(mol, dm)
    e_coul = np.einsum('ij,ji->', vj, dm).real * .5
    return e_coul

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def dipole_and_quadrupole(rho, grid, weights):
    dipole = np.empty(3)
    for i in range(3):
        dipole[i] = np.sum(rho * weights * grid[:, i], axis = 0)
    quadrupole = 3 * np.einsum('i, i, ij, ik -> jk', rho, weights, grid, grid)
    quad_diag = np.sum(rho * weights * np.sum(grid * grid, axis = 1))
    for i in range(3):
        quadrupole[i, i] = quadrupole[i, i] - quad_diag
    return dipole, quadrupole

gridLevel = 5
mode = 'hirshfeld'
do_cc = True

results_dir = 'results/'
if not os.path.exists(results_dir):
   os.makedirs(results_dir)

restricted = True

if len(sys.argv) != 4:
    print("""Provide three arguments, first is xyz filename of molecule, second is total charge of molecule, third is basiss""")
    quit()
xyzFilename = sys.argv[1]
totalCharge = float(sys.argv[2])

results_fname = os.path.split(xyzFilename)[1]
results_fname = os.path.splitext(results_fname)[0]
results_fname = "%s%s_%s.json"%(results_dir, results_fname, sys.argv[3])
print("results", results_fname)

mol = pyscf.gto.Mole()
mol.atom = xyzFilename
# mol.basis = 'sto-3g'

basis = {
    'H': sys.argv[3],
    'He': sys.argv[3],
    'Li': 'augccpvqz',
    'Be': 'augccpvqz',
    'B': sys.argv[3],
    'C': sys.argv[3],
    'N': sys.argv[3],
    'O': sys.argv[3],
    'F': sys.argv[3],
    'Ne': sys.argv[3],
    'Na': 'augccpvqz',
    'Mg': 'augccpvqz',
    'Al': sys.argv[3],
    'Si': sys.argv[3],
    'P': sys.argv[3],
    'S': sys.argv[3],
    'Cl': sys.argv[3],
    'Ar': sys.argv[3],
}

mol.basis = sys.argv[3]
mol.symmetry = False
mol.charge = int(totalCharge)
temp_ats = read(xyzFilename)
n_elec, core_elec, val_elec = electronCounter.countElectrons_symb(temp_ats.get_chemical_symbols())
core_elec = 0

mol.spin = int((n_elec - totalCharge) % 2)
print(mol.charge, mol.spin)
mol.build()


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
    if not restricted:
        dm_dft = dm_dft[0, :, :] + dm_dft[1, :, :]
    e_elec = getElectricEnergy(dft_res, mol, dm_dft)
    charges = Partitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel, fastMethod = False)
    sys.stdout.flush()
    return e_pot, charges, e_elec, dm_dft

results = dict()
settings = dict()
functionals = ['svwn', 'pbe', 'pbe0', 'rpbe', 'scan', 'rscan', 'r2scan', 'SCAN0,SCAN', 'blyp', 'b3lyp']
# functionals = ['scan']

settings['basisset'] = mol.basis
settings['restricted'] = restricted
settings['total_charge'] = mol.charge
settings['spin'] = mol.spin
settings['gridlevel'] = gridLevel
settings['xyz'] = os.path.abspath(xyzFilename)


etot_s = 'etot'
charges_s = 'charges'
e_coul_s = 'e_coulomb'
dm_s = 'density_matrix'
coul_diff = 'e_coulomb_diff'
charge_diff_energy = 'charge_diff_energy'
abs_charge_diff_int = 'abs_charge_diff_int'
squared_charge_diff_int = 'squared_charge_diff_int' 
occ_s = 'orb_occupancies'
orb_energies_s = 'orb_energies'

for functional in functionals:
    print("Start %s calculation"%functional)
    results[functional] = dict()
    e_tot, charges, e_coul, dm_dft = DFT_charges(mol, functional, restricted, gridLevel)
    results[functional][etot_s] = e_tot
    results[functional][charges_s] = charges
    results[functional][e_coul_s] = e_coul
    results[functional][dm_s] = dm_dft
    print('calculation of function %s done'%functional)
    print('coulomb energy', e_coul)
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
# _, e_coul = mf.energy_elec(dm_hf)
e_coul = getElectricEnergy(mf, mol, dm_hf)
print('coulomb energy', e_coul)
print('sum of hf charges', np.sum(charges_hf), '\n\n')
sys.stdout.flush()

results['hf'] = dict()
results['hf'][charges_s] = charges_hf
results['hf'][etot_s] = e_hf
results['hf'][e_coul_s] = e_coul
results['hf'][dm_s] = dm_hf
# e_dc_dft = dc_dft.get_dc_energy(mol, mf, False)
# results['hf']['df_dft'] = e_dc_dft

if do_cc:
    # Coupled Cluster calculation
    print("Start coupled cluster calculation")
    mycc = cc.CCSD(mf, frozen=core_elec)
    # mycc.async_io = False
    mycc.direct = True
    mycc.incore_complete = True
    mycc.run()
    e_cc = mycc.e_tot + mycc.ccsd_t()
    # print(e_cc, e_dc_dft)
    print('ccsd(t) energy', e_cc)

    dm_cc = mycc.make_rdm1(ao_repr=True)
    if not restricted:
        dm_cc = dm_cc[0] + dm_cc[1]
    sys.stdout.flush()
    charges_cc = Partitioning.getAtomicCharges(mol, dm_cc, mode, gridLevel)
    e_coul = getElectricEnergy(mf, mol, dm_cc)
    print('coulomb energy', e_coul)
    print('sum of cc charges', np.sum(charges_cc), '\n\n')
    sys.stdout.flush()

    results['cc'] = dict()
    results['cc'][charges_s] = charges_cc
    results['cc'][etot_s] = e_cc
    results['cc'][e_coul_s] = e_coul
    results['cc'][dm_s] = dm_cc

    ediff = getElectricEnergy(mf, mol, dm_hf - dm_cc)

    rho_cc, grid = Partitioning.getRho(mol, dm_cc, gridLevel)

    functionals.append('hf')
    functionals.insert(0, 'cc')

    for functional in functionals:
        dft_res = mol.RHF()
        results[functional][coul_diff] = results[functional][e_coul_s] - results['cc'][e_coul_s]
        temp = getElectricEnergy(dft_res, mol, results[functional][dm_s] - results['cc'][dm_s])
        results[functional][charge_diff_energy] = temp
        print('%s ecdiff, e_edif'%functional, results[functional][charge_diff_energy], results[functional][coul_diff])
        rho, grid = Partitioning.getRho(mol, results[functional][dm_s], gridLevel)
        results[functional][abs_charge_diff_int] = np.sum(np.abs( rho - rho_cc )* grid.weights)
        results[functional][squared_charge_diff_int] = np.sum((rho - rho_cc)**2 * grid.weights)
        print('%s norm abs, norm square'%functional, results[functional][abs_charge_diff_int], results[functional][squared_charge_diff_int])
        dipole, quadrupole = dipole_and_quadrupole(rho, grid.coords, grid.weights)
        results[functional]['dipole'] = dipole
        results[functional]['dipole_diff'] = np.linalg.norm(dipole - results['cc']['dipole'])
        results[functional]['quadrupole'] = quadrupole
        results[functional]['quadrupole_diff'] = np.linalg.norm(quadrupole - results['cc']['quadrupole'])
        results[functional]['rhodiff'] = np.max(rho - rho_cc)
        results[functional]['rhodiff_negative'] = np.max(rho_cc - rho)
        print("%s dipole moment"%functional, dipole)
        print("quadrupole", quadrupole, '\n')


summary = dict()
summary['settings'] = settings
summary['results'] = results
print('\n')
with open(results_fname, "w") as f:
    json.dump(summary, f, indent=4, sort_keys=True, cls = NumpyArrayEncoder)