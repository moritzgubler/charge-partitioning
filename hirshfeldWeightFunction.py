import pyscf
import numpy as np
import periodictable
import scipy.interpolate
import scipy.integrate
# import matplotlib.pyplot as plt


def hirshfeldPartitioningWeights(x : np.array(3), molecule: pyscf.gto.Mole, atomIndex):
    nAtoms = len(molecule._atom)


def getLogNormalizer(x, molecule: pyscf.gto.Mole):
    x = np.matrix(x)
    distances = np.zeros((x.shape[0], len(molecule._atom)))
    for i, atom in enumerate(molecule._atom):
        atomSymbol = atom[0]
        pos = atom[1]
        protonCount = periodictable.getNumberFromElementSymbol(atomSymbol)
    return scipy.special.logsumexp(distances, axis=1)


def getDensityFileName(protonCount: int):
    return '/kernph/gubmor00/git/charge-partitioning/radialAtomicDensities/byNumber/{:03d}'.format(protonCount) + '-density.AE'


def readDensity(fileName: str):
    f = open(fileName, mode='r')
    f.readline()
    r, rho = np.array([[float(x) for x in l.split()] for l in f]).transpose()
    return r, rho

def interpolateDensity(protonCount: int):
    r, rho = readDensity(getDensityFileName(protonCount))
    f_temp = scipy.interpolate.interp1d(r, rho, kind='cubic', bounds_error=False, fill_value='extrapolate') 
    r0 = np.zeros(len(r)+1)
    r0[0] = 0
    r0[1:] = r
    rho0 = np.zeros(len(r) + 1)
    rho0[0] = f_temp(0)
    rho0[1:] = rho
    f_temp = scipy.interpolate.interp1d(r0, rho0, kind='cubic', bounds_error=False, fill_value=0) 
    return lambda x: abs(f_temp(x)) / 4 / np.pi

def interpolateLogDensity(protonCount: int):
    r, rho = readDensity(getDensityFileName(protonCount))
    r0 = []
    rho0 = []
    rho = rho / 4 / np.pi
    for i in range(len(r)):
        if rho[i] > 0.0:
            r0.append(r[i])
            rho0.append(rho[i])
    rho0 = np.log(rho0)
    return scipy.interpolate.interp1d(r0, rho0, kind='cubic', bounds_error=False, fill_value='extrapolate') 


def createDensityInterpolationDictionary(molecule: pyscf.gto.Mole):
    functionDict = {}
    for atom in molecule._atom:
        atomSymbol = atom[0]
        protonCount = periodictable.getNumberFromElementSymbol(atomSymbol)
        if not atomSymbol in functionDict:
            functionDict[atomSymbol] = interpolateLogDensity(protonCount)
    return functionDict


def getNormalizer(x, molecule: pyscf.gto.Mole, functionDict: dict):
    x = np.matrix(x)
    distances = np.zeros((x.shape[0], len(molecule._atom)))
    for i, atom in enumerate(molecule._atom):
        atomSymbol = atom[0]
        pos = atom[1]
        distances[:,i] = functionDict[atomSymbol](np.linalg.norm(x - pos, axis=1))
    return scipy.special.logsumexp(distances, axis=1)
    # return np.sum(distances, axis=1)


def partitioningWeights(x : np.array(3), molecule: pyscf.gto.Mole, atomIndex, functionDict: dict):
    if not molecule.unit =="B":
        print("""Molecule must have Bohr units when Hirshfeld partitioning is used. 
         The molecule provided here has other units, proceed with care""")
    normalizer = getNormalizer(x, molecule, functionDict)
    atomSymb = molecule._atom[atomIndex][0]
    pos = molecule._atom[atomIndex][1]
    return np.exp( functionDict[atomSymb](np.linalg.norm(pos - x, axis=1)) - normalizer )
    # return (functionDict[atomSymb](np.linalg.norm(pos - x, axis=1))) / (normalizer + 1e-13)

if __name__ == '__main__':
    mol = pyscf.gto.Mole()
    mol.atom = '''O 0 0 0; H  0 0 -3; H 0 0 3'''
    mol.unit = "B"
    mol.basis = 'sto-3g'
    mol.build()

    functionDict = createDensityInterpolationDictionary(mol)
    N = 500
    x = np.zeros((N, 3))
    x[:, 2] = np.linspace(-50, 50, N)
    # x = np.zeros((1, 3))
    # x[0, 2] = 6
    w1 = partitioningWeights(x, mol, 0, functionDict)
    plt.plot(x[:, 2], w1)
    w2 = partitioningWeights(x, mol, 1, functionDict)
    plt.plot(x[:, 2], w2)
    w3 = partitioningWeights(x, mol, 2, functionDict)
    plt.plot(x[:, 2], w3)
    plt.show()

    quit()

    f = interpolateLogDensity(14)
    print('asdf')
    x = np.linspace(0, 200, 100)
    rho = f(x)

    int = scipy.integrate.quad(lambda x:  4 * np.pi * x**2 * np.exp(f(x)), 0, np.Infinity )[0]
    print(int)
    plt.plot(x, (rho))
    plt.show()
    print(rho)
