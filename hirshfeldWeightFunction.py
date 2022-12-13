import pyscf
import numpy as np
import periodictable
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt


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

def createAtomicDensitiesFromFiles(molecule: pyscf.gto.Mole):
        for atom in molecule._atom:
            atomSymbol = atom[0]
            protonCount = periodictable.getNumberFromElementSymbol(atomSymbol)


def getDensityFileName(protonCount: int):
    return 'radialAtomicDensities/byNumber/{:03d}'.format(protonCount) + '-density.AE'


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

if __name__ == '__main__':
    f = interpolateDensity(14)
    print('asdf')
    x = np.linspace(0, 50, 100)
    rho = f(x)

    int = scipy.integrate.quad(lambda x:  4 * np.pi * x**2 * f(x), 0, np.Infinity )[0]
    print(int)
    plt.plot(x, np.log(rho))
    plt.show()
    print(rho)