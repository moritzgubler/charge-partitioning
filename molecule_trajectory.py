import pyscf
import numpy as np

def moleculeTrajectory(moleculeStart: pyscf.gto.Mole, moleculeEnd: pyscf.gto.Mole, NSlices):
    posA = moleculeStart._atom[1]
    posB = moleculeEnd._atom[1]
    direction = posB - posA
    mt = []
    basis = moleculeStart.basis
    atomNames = moleculeStart._atom[0]

    for i in range(NSlices):
        molTemp = pyscf.gto.Mole()
        molTemp.basis = basis
        molTemp.atom = [atomNames, posA + direction * i / (NSlices -1)]
        molTemp.build()
        mt.append(molTemp)
    
    return molTemp

