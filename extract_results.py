
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
    functionals = [
        "cc",
        'hf',
        'lda',
        'pbe',
        'rpbe',
        'blyp',
        'b3lyp',
        'pbe0',
        'scan',
        'rscan',
        'r2scan',
        'SCAN0,SCAN',
    ]
    # get charges
    charges = get_key_result(dict_list, 'charges')
    rho_diff_energy = get_key_result(dict_list, 'charge_diff_energy')
    print('\ncharge diff energy')
    for fun in functionals:
        print(fun, np.mean(rho_diff_energy[fun]))

    print("\n maximal charge diff energy")
    for fun in functionals:
        print(fun, np.max(np.abs(rho_diff_energy[fun])))


    print('\n integral over squared charge difference')
    sqdiff = get_key_result(dict_list, 'squared_charge_diff_int')
    for fun in functionals:
        print(fun, np.mean(sqdiff[fun]))

    print('\n maximal squared charge difference')
    for fun in functionals:
        print(fun, np.max(sqdiff[fun]))

    rhodiff = get_key_result(dict_list, 'rhodiff')
    rhodiff_neg = get_key_result(dict_list, 'rhodiff_negative')
    print('\naverage maximal point wise charge difference')
    for fun in functionals:
        print(fun, np.mean(np.maximum(np.abs(rhodiff[fun]), np.abs(rhodiff_neg[fun]))))

    print('\nmaximal point wise charge difference')
    for fun in functionals:
        print(fun, np.max(np.maximum(np.abs(rhodiff[fun]), np.abs(rhodiff_neg[fun]))))

    charge_diff = dict()
    for fun in functionals:
        charge_diff[fun] = np.hstack(charges[fun]) - np.hstack(charges['cc'])

    print('\nrmse hirshfeld charges')
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

    print("\naverage dipole error:")
    for fun in functionals:
        print(fun, np.mean(dip_diff[fun]))

    print("\nmaximal dipole error:")
    for fun in functionals:
        print(fun, np.max(dip_diff[fun]))
    
    print("\naverage quadrupole error:")
    for fun in functionals:
        print(fun, np.mean(quad_diff[fun]))
    
    print("\nmaximal quadrupole error:")
    for fun in functionals:
        print(fun, np.max(quad_diff[fun]))

    # make energy plot:
    energies = get_key_result(dict_list, 'etot')
    # energies['dc_dft'] = []
    # for dictionary in dict_list:
    #     energies['dc_dft'].append(dictionary['results']['hf']['df_dft'])

    f_copy = functionals.copy()
    f_copy.remove('cc')
    # f_copy.append('dc_dft')
    # f_copy = ['scan', 'SCAN0,SCAN', 'b3lyp']
    # for fun in f_copy:
    #     plt.scatter(energies['cc'], energies[fun], label = fun)
    # plt.legend()
    # plt.xlabel('cc energies')
    # plt.ylabel('energy')
    # plt.title('energy correlation')
    # plt.grid()
    # plt.savefig('energy_correlation.pdf')
    # plt.show()
    # for fun in f_copy:
    #     plt.scatter(energies['cc'], np.array(energies[fun]) - np.array(energies['cc']), label = fun)
    # plt.legend()
    # plt.xlabel('cc energies')
    # plt.ylabel('E_DFT - E_CC')
    # plt.title('energy error')
    # plt.grid()
    # plt.savefig('energy_error.pdf')
    # plt.show()

    # # f_copy.remove('dc_dft')

    # for fun in f_copy:
    #     plt.scatter(np.hstack(charges['cc']), np.hstack(charges[fun]) - np.hstack(charges['cc']), label = fun)
    # plt.grid()
    # plt.legend()
    # plt.xlabel('CC charges')
    # plt.ylabel('charge_dft - charge_cc')
    # plt.show()
    

def get_key_result(dict_list, key):
    results = dict()
    functionals = list(dict_list[0]['results'].keys())
    # print(functionals)
    for functional in functionals:
        results[functional] = []
    for dictionary in dict_list:
        for functional in functionals:
            results[functional].append(dictionary['results'][functional][key])
    return results

if __name__ == '__main__':
    main()

