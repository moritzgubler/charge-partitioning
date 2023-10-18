import pyscf
import chargePartitioning.periodictable as periodictable

def countElectrons(mol: pyscf.gto.Mole):
    symbs, coords = zip(*mol._atom)
    totalElectrons = 0
    totalCoreElectrons = 0
    totalValenceElectrons = 0
    for symb in symbs:
        coreElectrons, valenceElectrons = periodictable.getCoreAndValenceElectrons(symb)
        totalElectrons += coreElectrons + valenceElectrons
        totalCoreElectrons += coreElectrons
        totalValenceElectrons += valenceElectrons
    return totalElectrons, totalCoreElectrons, totalValenceElectrons
