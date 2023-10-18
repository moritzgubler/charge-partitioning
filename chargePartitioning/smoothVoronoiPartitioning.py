import chargePartitioning.periodictable as periodictable
import scipy
import pyscf
import numpy as np

covalentRadiusFactor = .6

def getCovalentRadius(atomSymbol):
    return periodictable.getCovalentRadiosFromString(atomSymbol) * covalentRadiusFactor


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