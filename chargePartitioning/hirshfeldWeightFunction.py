import pyscf
import numpy as np
import chargePartitioning.periodictable as periodictable
import scipy.interpolate
import scipy.integrate
# import matplotlib.pyplot as plt
import os

def getDensityFileName(protonCount: int):
    return os.path.join(os.path.dirname(__file__), 'radialAtomicDensities', 'byNumber', "%03d-density.AE"%(protonCount))

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

def createDensityInterpolationDictionary(elements: list):
    functionDict = {}
    for atomSymbol in elements:
        protonCount = periodictable.getNumberFromElementSymbol(atomSymbol)
        if not atomSymbol in functionDict:
            functionDict[atomSymbol] = interpolateLogDensity(protonCount)
    return functionDict

def getNormalizer(x, pos: np.array, elements: list, functionDict: dict, lat = None, cutoff_bohr = 10.0):

    if lat is not None:
        periodic = True
    else:
        periodic = False
    
    if not periodic:
        poscopy = np.copy(pos)
        element_copy = elements.copy()
    else:
        poscopy, element_copy = periodicePositions(lat, pos, elements, cutoff_bohr)


    x = np.matrix(x)
    distances = np.zeros((x.shape[0], poscopy.shape[0]))
    for i in range(poscopy.shape[0]):
        distances[:,i] = functionDict[element_copy[i]](np.linalg.norm(x - poscopy[i, :], axis=1))
    return scipy.special.logsumexp(distances, axis=1)

def countGostCellsFractional(lat, cutoff):
    a = lat[0, :]
    b = lat[1, :]
    c = lat[2, :]

    ab = np.cross(a, b)
    ab = ab / np.linalg.norm(ab)
    ac = np.cross(a, c)
    ac = ac / np.linalg.norm(ac)
    bc = np.cross(b, c)
    bc = bc / np.linalg.norm(bc)

    proj_a = np.abs( np.sum(a * bc ) )
    proj_b = np.abs( np.sum(b * ac ) )
    proj_c = np.abs( np.sum(c * ab ) )

    return np.array([cutoff / proj_a, cutoff / proj_b, cutoff / proj_c])

def periodicePositions(lat, pos, elements, cutoff):
    nat = pos.shape[0]
    lens = countGostCellsFractional(lat, cutoff)
    frac_lens = np.ceil(lens).astype('int')

    a_inv = np.linalg.inv(lat)

    red = np.zeros(pos.shape)

    for i in range(pos.shape[0]):
        red[i, :] =  pos[i, :] @ a_inv # = (a_inv.T @ pos[i, :].T).T

    images = []
    element_images = []
    for iat in range(pos.shape[0]):
        for i in range(-frac_lens[0], frac_lens[0] + 1):
            for j in range(-frac_lens[1], frac_lens[1] + 1):
                for k in range(-frac_lens[2], frac_lens[2] + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    lat_shift = red[iat, :] + np.array([i, j, k])
                    accept_i = i ==0 or abs(lat_shift[0]) < lens[0] or abs(lat_shift[0] - 1) < lens[0]
                    accept_j = j ==0 or abs(lat_shift[1]) < lens[1] or abs(lat_shift[1] - 1) < lens[1]
                    accept_k = k ==0 or abs(lat_shift[2]) < lens[2] or abs(lat_shift[2] - 1) < lens[2]
                    if accept_i and accept_j and accept_k:
                        images.append(lat.T @ lat_shift)
                        element_images.append(elements[iat])

    allnat = nat + len(element_images)
    all_elements = elements + element_images
    all_pos = np.zeros((allnat, 3))
    all_pos[:nat, :] = pos
    for i in range(nat, allnat):
        all_pos[i, :] = images[i-nat]
    return all_pos, all_elements



def partitioningWeights(x : np.array(3), atom_position: np. array, atomic_density_function, normalizer: np.array, lat = None, cutoff_bohr = 10.0):
    """
    x: (gridlen, 3)
    atompos: 3
    atomelement: str
    atomic_density_function: function
    normalizer (gridlen), construct with getNormalizer
    """
    if lat is not None:
        periodic = True
    else:
        periodic = False
    
    # weights = np.empty(x.shape[0])
    weights = np.exp( atomic_density_function(np.linalg.norm(atom_position - x, axis=1)) - normalizer )

    if periodic:
        p_images, elem_images = periodicePositions(lat, np.reshape(atom_position, (1, 3)), ['C'], cutoff_bohr)
        for i in range(1, len(elem_images)):
            weights = weights + np.exp(atomic_density_function(np.linalg.norm(p_images[i, :] - x, axis=1)) - normalizer)

    # return np.exp( atomic_density_function(np.linalg.norm(atom_position - x, axis=1)) - normalizer )
    return weights

def partitionCharges(x_grid, rho_grid, positions, elements):
    functionDict = createDensityInterpolationDictionary(elements)
    normalizer = getNormalizer(x_grid, positions, elements, functionDict)
    charges = []
    for i in range(len(elements)):
        partition_weights = partitioningWeights(x_grid, positions[i, :], functionDict[elements[i]], normalizer)
        charges.append(np.sum(partition_weights * rho_grid) / len(rho_grid))
    return charges
    

def partitioningWeights_molecule(x : np.array(3), molecule: pyscf.gto.Mole, atomIndex, functionDict: dict):
    pos = molecule._atom[atomIndex][1]
    all_pos = np.zeros((len(molecule._atom), 3))
    elements = []
    for i in range(len(molecule._atom)):
        all_pos[i, :] = molecule._atom[i][1]
        elements.append(molecule._atom[i][0])
    atom_function = functionDict[molecule._atom[atomIndex][0]]
    normalizer = getNormalizer(x, all_pos, elements, functionDict)
    return partitioningWeights(x, pos, atom_function, normalizer)


if __name__ == '__main__':
    mol = pyscf.gto.Mole()
    mol.atom = '''O 0 0 0; H  0 0 -3; H 0 0 3'''
    # mol.unit = "B"
    mol.basis = 'sto-3g'
    mol.build()

    elements = []
    pos = np.zeros((len(mol._atom), 3))
    print(elements, len(mol._atom))
    i = 0
    for atom in mol._atom:
        print(atom[0])
        elements.append(atom[0])
        pos[i, :] = mol._atom[i][1]
        i += 1

    functionDict = createDensityInterpolationDictionary(elements)
    N = 500
    x = np.zeros((N, 3))
    x[:, 2] = np.linspace(-50, 50, N)
    # x = np.zeros((1, 3))
    # x[0, 2] = 6
    import matplotlib.pyplot as plt
    normalizer = getNormalizer(x, pos, elements, functionDict)
    w1 = partitioningWeights_molecule(x, mol, 0, functionDict)
    w11 = partitioningWeights(x, mol._atom[0][1], functionDict[mol._atom[0][0]], normalizer)
    plt.plot(x[:, 2], w1, x[:, 2], w11)
    plt.show()
    w2 = partitioningWeights_molecule(x, mol, 1, functionDict)
    plt.plot(x[:, 2], w2)
    w3 = partitioningWeights_molecule(x, mol, 2, functionDict)
    plt.plot(x[:, 2], w3)
    plt.show()

    from ase.atoms import Atoms
    from ase.io import write
    from ase.lattice.cubic import Diamond

    ats = Diamond('Si')

    new_pos, new_elements = periodicePositions(ats.cell, ats.get_positions(), ats.get_chemical_symbols(), 5)

    newAts = Atoms(new_elements, new_pos)
    write('bulk.extxyz', ats)
    write('clust.xyz', newAts)

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
