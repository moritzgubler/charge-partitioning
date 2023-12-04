
import argparse
import numpy as np
import json
from ase.io import read, write
from ase.atoms import Atoms
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
    args = parser.parse_args()

    file_list = args.file

    dict_list = []

    for file in file_list:
        with open(file.name, mode='r') as f:
            dict_list.append(json.load(f))
    
    # atom_list = []
    # for dictionary in dict_list:
    #     atom_list.append(read(dictionary['settings']['xyz']))
    
    functionals = list(dict_list[0]['results'].keys())
    # get charges
    charges = get_key_result(dict_list, 'charges')
    rho_diff_energy = get_key_result(dict_list, 'charge_diff_energy')
    print('\ncharge diff energy')
    for fun in functionals:
        print(fun, np.mean(rho_diff_energy[fun]))


    print('\n integral over squared charge difference')
    sqdiff = get_key_result(dict_list, 'squared_charge_diff_int')
    for fun in functionals:
        print(fun, np.mean(sqdiff[fun]))

    charge_diff = dict()
    for fun in functionals:
        charge_diff[fun] = np.hstack(charges[fun]) - np.hstack(charges['cc'])

    print('\nrmse e rho diff (hirshfeld charges)')
    n_charges = len(charge_diff['cc'])
    for fun in functionals:
        print(fun, 1/n_charges * np.linalg.norm(charge_diff[fun]))
    
    print('\nmaxdiff of hirshfeld charges')
    for fun in functionals:
        print(fun, np.max(np.abs(charge_diff[fun])))

    coul_diff = get_key_result(dict_list, 'e_coulomb_diff')
    print('\n rmse of coulomb energy')
    for fun in functionals:
        print(fun, np.linalg.norm(coul_diff[fun]) / len(coul_diff['cc']))
    print('\nmaximal difference of coulomb energy')
    for fun in functionals:
        print(fun, np.max(np.abs(coul_diff[fun])))

    dip_diff = get_key_result(dict_list, 'dipole_diff')
    quad_diff = get_key_result(dict_list, 'quadrupole_diff')

    print("average dipole error:")
    for fun in functionals:
        print(fun, np.mean(dip_diff[fun]))
    print("average quadrupole error:")
    for fun in functionals:
        print(fun, np.mean(quad_diff[fun]))
    

def get_key_result(dict_list, key):
    results = dict()
    functionals = list(dict_list[0]['results'].keys())
    print(functionals)
    for functional in functionals:
        results[functional] = []
    for dictionary in dict_list:
        for functional in functionals:
            results[functional].append(dictionary['results'][functional][key])
    return results

if __name__ == '__main__':
    main()

