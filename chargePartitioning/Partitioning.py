import pyscf
import numpy as np
import chargePartitioning.periodictable as periodictable
import scipy
import chargePartitioning.hirshfeldWeightFunction as hirshfeldWeightFunction
import chargePartitioning.smoothVoronoiPartitioning as smoothVoronoiPartitioning
from pyscf import dft



def createGrid(molecule: pyscf.gto.Mole, gridLevel=4):
    grid = pyscf.dft.gen_grid.Grids(molecule)
    grid.level = gridLevel
    grid.build()
    return grid


def integrateDensityOfAtom(molecule: pyscf.gto.Mole, rho: np.array, atomIndex: int, grid: pyscf.dft.gen_grid.Grids, mode='hirshfeld'):
    if mode == 'hirshfeld':
        functionDict = hirshfeldWeightFunction.createDensityInterpolationDictionary(molecule.elements)
        weights = hirshfeldWeightFunction.partitioningWeights_molecule(grid.coords, molecule, atomIndex, functionDict)
    elif mode == 'voronoi':
        weights = smoothVoronoiPartitioning.partitioningWeights(grid.coords, molecule, atomIndex)
    else:
        print('Unkown mode: ', mode)
        quit()
    
    return np.einsum('i,i,i->', rho, grid.weights, weights)

def splitMoleculeToAtoms(molecule: pyscf.gto.Mole):
    atomList = []
    basis = molecule.basis
    for atom in molecule._atom:
        if periodictable.valenceElectronsDict[atom[0]] % 2 == 0:
            spin = 0
        else:
            spin = 1
        atomList.append(pyscf.gto.M(atom=[atom], basis=basis, spin = spin, unit='B'))
        # atomList[-1].build()
    return atomList

def getAtomicCharges(molecule: pyscf.gto.Mole, densitymatrix,  mode='hirshfeld', gridLevel = 3, fastMethod = True):
    if not fastMethod:
        grid = createGrid(molecule, gridLevel=gridLevel)
        print('number of gridpoints used for charge partitioning', grid.coords.shape[0])
        ao = molecule.eval_gto(eval_name='GTOval', coords=grid.coords)
        if type(densitymatrix) is tuple:
            rhoA = np.einsum('pi,ij,pj->p', ao, densitymatrix[0], ao)
            rhoB = np.einsum('pi,ij,pj->p', ao, densitymatrix[1], ao)
            rho = rhoA + rhoB
        else:
            rho = np.einsum('pi,ij,pj->p', ao, densitymatrix, ao)
    
    charges = []
    if fastMethod:
        atomList = splitMoleculeToAtoms(molecule)
    for i in range(len(molecule._atom)):
        atomSymb = molecule._atom[i][0]
        if fastMethod:
            grid = createGrid(atomList[i], gridLevel)
            print('number of gridpoints used for charge partitioning', grid.coords.shape[0])
            ao = molecule.eval_gto(eval_name='GTOval', coords=grid.coords)
            if type(densitymatrix) is tuple or len(densitymatrix.shape) == 3:
                rhoA = np.einsum('pi,ij,pj->p', ao, densitymatrix[0], ao)
                rhoB = np.einsum('pi,ij,pj->p', ao, densitymatrix[1], ao)
                rho = rhoA + rhoB
            else:
                rho = np.einsum('pi,ij,pj->p', ao, densitymatrix, ao)
        protonCount = periodictable.getNumberFromElementSymbol(atomSymb)
        charges.append(-integrateDensityOfAtom(molecule, rho, i, grid, mode) + protonCount)
    return charges

