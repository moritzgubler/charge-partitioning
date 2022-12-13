import pyscf
import numpy as np
import periodictable
import scipy
import hirshfeldWeightFunction

covalentRadiusFactor = .6

def getCovalentRadius(atomSymbol):
    return periodictable.getCovalentRadiosFromString(atomSymbol) * covalentRadiusFactor

# def splitMoleculeToAtoms(molecule: pyscf.gto.Mole):
#     atomList = []
#     basis = molecule.basis
#     for atom in molecule._atom:
#         atomList.append(pyscf.gto.Mole(atom=atom, basis=basis))

def createGrid(molecule: pyscf.gto.Mole, gridLevel=4):
    grid = pyscf.dft.gen_grid.Grids(molecule)
    grid.level = gridLevel
    grid.build()
    return grid

def getLogNormalizer(x, molecule: pyscf.gto.Mole):
    x = np.matrix(x)
    distances = np.zeros((x.shape[0], len(molecule._atom)))
    for i, atom in enumerate(molecule._atom):
        atomSymbol = atom[0]
        pos = atom[1]
        covalentRadius = getCovalentRadius(atomSymbol)
        distances[:,i] = -(np.linalg.norm(x - pos, axis=1)**2) / (2 * covalentRadius**2)
    return scipy.special.logsumexp(distances, axis=1)


def partitioningWeights(x : np.array(3), molecule: pyscf.gto.Mole, atomIndex):
    nAtoms = len(molecule._atom)
    normalizer = getLogNormalizer(x, molecule)
    covalentRadius = getCovalentRadius(molecule._atom[atomIndex][0])
    pos = molecule._atom[atomIndex][1]
    return np.exp( -np.linalg.norm(pos - x, axis=1)**2 /(2 * covalentRadius**2) - normalizer )


def integrateDensityOfAtom(molecule: pyscf.gto.Mole, dm, atomIndex):

    # basis = molecule.basis
    # temp_atom = pyscf.gto.Mole()
    # temp_atom.atom = [molecule._atom[atomIndex]]
    # temp_atom.basis = basis
    # temp_atom.spin = 1
    # temp_atom.build()
    
    functionDict = hirshfeldWeightFunction.createDensityInterpolationDictionary(molecule)

    grid = createGrid(molecule, gridLevel=3)
    ao = molecule.eval_gto(eval_name='GTOval', coords=grid.coords)
    # check if calculation was restricted or unrestricted:
    if type(dm) is tuple:
        rhoA = np.einsum('pi,ij,pj->p', ao, dm[0], ao)
        rhoB = np.einsum('pi,ij,pj->p', ao, dm[1], ao)
        rho = rhoA + rhoB
    else:
        rho = np.einsum('pi,ij,pj->p', ao, dm, ao)

    weights = hirshfeldWeightFunction.partitioningWeights(grid.coords, molecule, atomIndex, functionDict)
    return np.einsum('i,i,i->', rho, grid.weights, weights)

def getAtomicCharges(molecule: pyscf.gto.Mole, dm):
    charges = []
    for i in range(len(molecule._atom)):
        charges.append(integrateDensityOfAtom(molecule, dm, i))
    return charges

