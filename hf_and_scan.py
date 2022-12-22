from pyscf import gto, dft
import pyscf
import numpy
mol = pyscf.gto.Mole()
mol.atom = '''H  0 0 0; H 0 0 1.1'''
mol.charge = 0
mol.basis = 'ccpv5z'
mol.build()

grids = dft.gen_grid.Grids(mol)

# grids.coords = numpy.random.random((100,3))  # 100 random points
# grids.weights = numpy.random.random(100)
# nao = mol.nao_nr()
# dm = numpy.random.random((2,nao,nao))

mf = mol.RHF().run()

ecoul, eelec = mf.energy_elec()
ecoul = ecoul - eelec
exh = -(mf.e_tot - ecoul -eelec)
print(ecoul, eelec, ecoul + eelec, mf.e_tot - ecoul -eelec)

dm_hf = mf.make_rdm1()

vj, vk = mf.get_jk(mol, dm_hf)
print('dm', dm_hf.shape)
print('vk', vk.shape)


# e_x = mf.e_tot - e_elec - e_coul
# print('ex', e_x)

# dft_res = dft.RKS(mol)
# dft_res.xc = 'HF'
# dft_res.newton()
# dft_res.kernel()
# dm_dft = dft_res.make_rdm1(ao_repr=True)


grids = dft.gen_grid.Grids(mol)



# ao = mol.eval_gto(eval_name='GTOval', coords=grids.coords)
# rho = numpy.einsum('pi,ij,pj->p', ao, dm_hf, ao)
nao = mol.nao_nr()

mycc = mf.CCSD().run()

# et = mycc.ccsd_t()
# print('ccsd_t', mycc.e_tot + et)

ni = dft.numint.NumInt()

nelec, xc_energy, vxc = ni.get_vxc(mol=mol, grids=grids, xc_code='SCAN,SCAN', dms=dm_hf)

# nelec, hf_x_energy, vxc = ni.get_vxc(mol=mol, grids=grids, xc_code='HF,', dms=dm_dft)

print(mf.e_tot +xc_energy-exh, xc_energy, exh, xc_energy-exh)
quit()

nelec, exc, vxc = dft.numint.NumInt.get_vxc(mol=mol, grids=grids, xc_code='scan', dms=dm_hf)
print(exc)