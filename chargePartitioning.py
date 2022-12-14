import pyscf
import numpy as np
import periodictable
import scipy
import hirshfeldWeightFunction
import smoothVoronoiPartitioning
from pyscf import dft



def createGrid(molecule: pyscf.gto.Mole, gridLevel=4):
    grid = pyscf.dft.gen_grid.Grids(molecule)
    grid.level = gridLevel
    grid.build()
    return grid


def integrateDensityOfAtom(molecule: pyscf.gto.Mole, rho: np.array, atomIndex: int, grid: pyscf.dft.gen_grid.Grids, mode='hirshfeld'):
    if mode == 'hirshfeld':
        functionDict = hirshfeldWeightFunction.createDensityInterpolationDictionary(molecule)
        weights = hirshfeldWeightFunction.partitioningWeights(grid.coords, molecule, atomIndex, functionDict)
    elif mode == 'voronoi':
        weights = smoothVoronoiPartitioning.partitioningWeights(grid.coords, molecule, atomIndex)
    else:
        print('Unkown mode: ', mode)
        quit()
    
    return np.einsum('i,i,i->', rho, grid.weights, weights)

def getAtomicCharges(molecule: pyscf.gto.Mole, densitymatrix,  mode='hirshfeld', gridLevel = 3):
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
    for i in range(len(molecule._atom)):
        atomSymb = molecule._atom[i][0]
        protonCount = periodictable.getNumberFromElementSymbol(atomSymb)
        charges.append(-integrateDensityOfAtom(molecule, rho, i, grid, mode) + protonCount)
    return charges

