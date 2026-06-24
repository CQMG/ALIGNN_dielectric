#!/usr/bin/env python3

# Modified from https://github.com/usnistgov/alignn/blob/main/alignn/examples/sample_data/scripts/generate_sample_data_reg.py

import numpy as np
from jarvis.db.webpages import Webpage
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from jarvis.core.spectrum import Spectrum


new_dist = np.arange(0, 15, 0.05)

D3D = data("dft_3d")

valid_samples = []
for i in D3D:
    if i['mbj_bandgap'] != 'na' and float(i['mbj_bandgap']) > 0.15:
        valid_samples.append(i)

print("Pre-parsed valid insulators (" + str(len(valid_samples)) + ")")


for sample in valid_samples:

    if sample['jid'] == "JVASP-74601":
        w = Webpage(jid=sample['jid'])

        D = w.to_dict()['basic_info']

        mbj_dir_gap = float(D['main_optics_mbj']['main_optics_mbj_info']['opto_dir_gap'])
        mbj_indir_gap = float(D['main_optics_mbj']['main_optics_mbj_info']['opto_indir_gap'])

        ref_slme = float(D['main_optics_mbj']['main_optics_mbj_info']['solar_slme'])
        ref_sq = float(D['main_optics_mbj']['main_optics_mbj_info']['solar_sq'])

        print(sample['jid'] + ":")
        print(" Direct gap:   " + str(mbj_dir_gap))
        print(" Indirect gap: " + str(mbj_indir_gap))
        print(" SLME:         " + str(ref_slme))
        print(" SQ:           " + str(ref_sq))


'''
convergence_info
wannier_band_comparison
vacancy_formation_energy
raman_dat
main_relax_info
main_band
main_hse06_band
effective_mass
main_pbe0_band
main_optics_semilocal
main_optics_mbj
main_elastic
main_boltz
main_lepsilon_info
main_spillage_info
efg_raw_tensor
max_efg
max_efg_eta
main_stm_pos
main_stm_neg
'''




'''
jid
spg_number
spg_symbol
formula
formation_energy_peratom
func
optb88vdw_bandgap
atoms
#                                       slme
magmom_oszicar
spillage
elastic_tensor
effective_masses_300K
kpoint_length_unit
maxdiff_mesh
maxdiff_bz
encut
optb88vdw_total_energy
epsx
epsy
epsz
mepsx
mepsy
mepsz
modes
magmom_outcar
max_efg
avg_elec_mass
avg_hole_mass
icsd
dfpt_piezo_max_eij
dfpt_piezo_max_dij
dfpt_piezo_max_dielectric
dfpt_piezo_max_dielectric_electronic
dfpt_piezo_max_dielectric_ionic
max_ir_mode
min_ir_mode
n-Seebeck
p-Seebeck
n-powerfact
p-powerfact
ncond
pcond
nkappa
pkappa
ehull
Tc_supercon
dimensionality
efg
xml_data_link
typ
exfoliation_energy
spg
crys
density
poisson
raw_files
nat
bulk_modulus_kv
shear_modulus_gv
#                                       mbj_bandgap
hse_gap
reference
search
'''