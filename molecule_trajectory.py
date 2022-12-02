import pyscf
import numpy as np

def moleculeTrajectory(moleculeStart: pyscf.gto.Mole, moleculeEnd: pyscf.gto.Mole, NSlices):
    posA = moleculeStart.atom_coords()
    posB = moleculeEnd.atom_coords()
    moleculeStart.atom_coords
    direction = posB - posA
    mt = []
    basis = moleculeStart.basis
    atomNames = getAtomnames(moleculeStart)

    for i in range(NSlices):
        molTemp = pyscf.gto.Mole()
        molTemp.basis = basis
        molTemp.atom =  createAtom(posA + direction * i / (NSlices -1), atomNames)
        molTemp.build()
        mt.append(molTemp)
    
    return mt

def createAtom(pos, atomNames):
    atoms = []
    for i in range(len(atomNames)):
        atoms.append((atomNames[i], pos[i, :]))
    return atoms

def getAtomnames(molecule):
    atomNames = []
    for i in range(len(molecule._atom)):
        atomNames.append(molecule.atom_symbol(i))
    return atomNames
