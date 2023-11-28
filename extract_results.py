
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
    
    atom_list = []
    for dictionary in dict_list:
        atom_list.append(read(dictionary['settings']['xyz']))
    
    functionals = list(dict_list[0]['results'].keys())
    # get charges
    charges = get_key_result(dict_list, 'charges')
    

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

