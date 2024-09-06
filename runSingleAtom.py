
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

rgrid = np.array(["0.97578195523695E-18", "0.29483069282768E-04", "0.61082173640461E-04", "0.94949178290611E-04", "0.13124684686428E-03", "0.17014962340302E-03", "0.21184447044981E-03", "0.25653176725507E-03", "0.30442627240692E-03", "0.35575815550320E-03", "0.41077410281323E-03", "0.46973850223137E-03", "0.53293471320370E-03", "0.60066642771546E-03", "0.67325912886222E-03", "0.75106165399386E-03", "0.83444786992019E-03", "0.92381846820159E-03", "0.10196028891215E-02", "0.11222613835509E-02", "0.12322872225722E-02", "0.13502090654347E-02", "0.14765934971652E-02", "0.16120477479678E-02", "0.17572226074079E-02", "0.19128155473023E-02", "0.20795740682274E-02", "0.22582992856177E-02", "0.24498497725618E-02", "0.26551456776172E-02", "0.28751731372654E-02", "0.31109890040199E-02", "0.33637259126873E-02", "0.36345977088734E-02", "0.39249052655295E-02", "0.42360427151559E-02", "0.45695041272231E-02", "0.49268906624595E-02", "0.53099182378730E-02", "0.57204257387563E-02", "0.61603838164655E-02", "0.66319043134761E-02", "0.71372503601200E-02", "0.76788471905016E-02", "0.82592937283951E-02", "0.88813749974526E-02", "0.95480754138094E-02", "0.10262593023187E-01", "0.11028354748868E-01", "0.11849032721462E-01", "0.12728561766261E-01", "0.13671158129094E-01", "0.14681339527142E-01", "0.15763946616981E-01", "0.16924165978315E-01", "0.18167554718499E-01", "0.19500066809896E-01", "0.20928081279553E-01", "0.22458432378533E-01", "0.24098441866553E-01", "0.25855953556357E-01", "0.27739370271550E-01", "0.29757693381404E-01", "0.31920565086455E-01", "0.34238313639551E-01", "0.36722001698414E-01", "0.39383478017693E-01", "0.42235432700963E-01", "0.45291456246181E-01", "0.48566102631601E-01", "0.52074956703276E-01", "0.55834706139791E-01", "0.59863218284852E-01", "0.64179622153785E-01", "0.68804395935657E-01", "0.73759460328728E-01", "0.79068278063012E-01", "0.84755959979821E-01", "0.90849378054143E-01", "0.97377285761297E-01", "0.10437044620442E+00", "0.11186176843355E+00", "0.11988645240026E+00", "0.12848214300340E+00", "0.13768909369134E+00", "0.14755034009334E+00", "0.15811188415740E+00", "0.16942288927237E+00", "0.18153588684917E+00", "0.19450699482699E+00", "0.20839614855590E+00", "0.22326734448556E+00", "0.23918889705942E+00", "0.25623370917446E+00", "0.27447955651526E+00", "0.29400938600797E+00", "0.31491162856155E+00", "0.33728052616913E+00", "0.36121647332923E+00", "0.38682637261280E+00", "0.41422400404472E+00", "0.44353040778554E+00", "0.47487427938818E+00", "0.50839237666158E+00", "0.54422993689737E+00", "0.58254110290335E+00", "0.62348935593563E+00", "0.66724795322943E+00", "0.71400036739287E+00", "0.76394072444865E+00", "0.81727423678444E+00", "0.87421762670440E+00", "0.93499953566278E+00", "0.99986091361083E+00", "0.10690553822024E+01", "0.11428495648913E+01", "0.12215233762238E+01", "0.13053702618907E+01", "0.13946973803820E+01", "0.14898257163832E+01", "0.15910901154124E+01", "0.16988392286241E+01", "0.18134353562483E+01", "0.19352541778144E+01", "0.20646843571704E+01", "0.22021270103889E+01", "0.23479950249930E+01", "0.25027122195788E+01", "0.26667123339008E+01", "0.28404378408543E+01", "0.30243385735729E+01", "0.32188701630746E+01", "0.34244922845610E+01", "0.36416667135785E+01", "0.38708551967931E+01", "0.41125171460457E+01", "0.43671071685993E+01", "0.46350724509653E+01", "0.49168500183013E+01", "0.52128638959647E+01", "0.55235222042457E+01", "0.58492142214126E+01", "0.61903074538070E+01", "0.65471447546554E+01", "0.69200415353393E+01", "0.73092831139412E+01", "0.77151222458371E+01", "0.81377768798438E+01", "0.85774281809213E+01", "0.90342188566791E+01", "0.95082518200100E+01", "0.99995892142010E+01", "0.10508251820010E+02", "0.11034218856679E+02", "0.11577428180921E+02", "0.12137776879844E+02", "0.12715122245837E+02", "0.13309283113941E+02", "0.13920041535339E+02", "0.14547144754655E+02", "0.15190307453807E+02", "0.15849214221413E+02", "0.16523522204246E+02", "0.17212863895965E+02", "0.17916850018301E+02", "0.18635072450965E+02", "0.19367107168599E+02", "0.20112517146046E+02", "0.20870855196793E+02", "0.21641666713578E+02", "0.22424492284561E+02", "0.23218870163075E+02", "0.24024338573573E+02", "0.24840437840854E+02", "0.25666712333901E+02", "0.26502712219579E+02", "0.27347995024993E+02", "0.28202127010389E+02", "0.29064684357170E+02", "0.29935254177814E+02", "0.30813435356248E+02", "0.31698839228624E+02", "0.32591090115412E+02", "0.33489825716383E+02", "0.34394697380382E+02", "0.35305370261891E+02", "0.36221523376224E+02", "0.37142849564891E+02", "0.38069055382202E+02", "0.38999860913611E+02", "0.39934999535663E+02", "0.40874217626704E+02"])
grid = np.zeros((len(rgrid), 3))
grid[:, 0] = rgrid

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

restricted = False

if len(sys.argv) != 2:
    print("""Provide one argument, the basiss""")
    quit()
basis = sys.argv[1]
totalCharge = 1
name = 'Na'

results_fname = "%s_%s.json"%(name, basis)
print("results", results_fname)

mol = pyscf.gto.Mole()
# mol.basis = 'sto-3g'
mol.atom = "Na 0.0 0.0 0.0"
mol.basis = sys.argv[1]
mol.symmetry = False
mol.charge = totalCharge
core_elec = 0

mol.spin = 0
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
    # charges = Partitioning.getAtomicCharges(mol, dm_dft, mode, gridLevel, fastMethod = False)
    sys.stdout.flush()
    return e_pot, e_elec, dm_dft

results = dict()
settings = dict()
functionals = ['lda', 'pbe', 'pbe0', 'rpbe', 'scan', 'rscan', 'r2scan', 'SCAN0,SCAN', 'blyp', 'b3lyp']
# functionals = ['scan']

settings['basisset'] = mol.basis
settings['restricted'] = restricted
settings['total_charge'] = mol.charge
settings['spin'] = mol.spin
settings['gridlevel'] = gridLevel


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
    e_tot, e_coul, dm_dft = DFT_charges(mol, functional, restricted, gridLevel)
    ao = mol.eval_gto(eval_name='GTOval', coords=grid.coords)
    rho = np.einsum('pi,ij,pj->p', ao, dm_dft, ao)
    results[functional][rho] = rho
    results[functional][grid] = rgrid
    results[functional][etot_s] = e_tot
    results[functional][e_coul_s] = e_coul
    results[functional][dm_s] = dm_dft
    print('calculation of function %s done'%functional)
    print('coulomb energy', e_coul)
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
# _, e_coul = mf.energy_elec(dm_hf)
e_coul = getElectricEnergy(mf, mol, dm_hf)
print('coulomb energy', e_coul)
sys.stdout.flush()

results['hf'] = dict()
results['hf'][etot_s] = e_hf
results['hf'][e_coul_s] = e_coul
results['hf'][dm_s] = dm_hf
e_dc_dft = dc_dft.get_dc_energy(mol, mf, False)
results['hf']['df_dft'] = e_dc_dft

if do_cc:
    # Coupled Cluster calculation
    print("Start coupled cluster calculation")
    mycc = cc.CCSD(mf, frozen=core_elec)
    # mycc.async_io = False
    mycc.direct = True
    mycc.incore_complete = True
    mycc.run()
    e_cc = mycc.e_tot + mycc.ccsd_t()
    print(e_cc, e_dc_dft)

    dm_cc = mycc.make_rdm1(ao_repr=True)
    if not restricted:
        dm_cc = dm_cc[0] + dm_cc[1]
    sys.stdout.flush()
    e_coul = getElectricEnergy(mf, mol, dm_cc)
    print('coulomb energy', e_coul)
    sys.stdout.flush()

    results['cc'] = dict()
    results['cc'][etot_s] = e_cc
    results['cc'][e_coul_s] = e_coul
    results['cc'][dm_s] = dm_cc

    ediff = getElectricEnergy(mf, mol, dm_hf - dm_cc)

    rho_cc, grid = Partitioning.getRho(mol, dm_cc, gridLevel)

    functionals.append('hf')
    functionals.insert(0, 'cc')

    f = open('charges.dat', 'w')
    f.write("# " + str(functionals) + '\n')
    
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
        print("%s dipole moment"%functional, dipole)
        print("quadrupole", quadrupole, '\n')


summary = dict()
summary['settings'] = settings
summary['results'] = results
print('\n')
with open(results_fname, "w") as f:
    json.dump(summary, f, indent=4, sort_keys=True, cls = NumpyArrayEncoder)