# Modified from https://github.com/usnistgov/alignn/blob/main/alignn/examples/sample_data/scripts/generate_sample_data_reg.py

import numpy as np
from jarvis.db.webpages import Webpage
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from jarvis.core.spectrum import Spectrum


new_dist = np.arange(0, 15, 0.05)

SAMPLE_COUNT = int(input("How many samples to gather? : "))
D3D = data("dft_3d")

valid_samples = []
for i in D3D:
    if i['mbj_bandgap'] != 'na' and float(i['mbj_bandgap']) > 0.15:
        valid_samples.append(i)

print("Pre-parsed valid insulators (" + str(len(valid_samples)) + ")")

count = 0

print("Parsing and collecting samples:\t 0%")

# Progress info:
threshold_offset = 1
next_threshold = threshold_offset

# Rejection counter
reject_count = 0

lines = []
names = []

max_value = -500.0

for sample in valid_samples:
    w = Webpage(jid=sample['jid'])
    try:
        mbj_dielectric = w.get_dft_mbj_dielectric_function()
    except KeyError:
        print("Rejected " + sample['jid'] + ":\t Could not read dielectric function")
        reject_count += 1
        continue
    s = Spectrum(x=mbj_dielectric['energies'], y=mbj_dielectric['imag_xx'])
    interp = np.array(s.get_interpolated_values(new_dist=new_dist))

    max_value = max(max_value, interp.max())

    if np.isnan(interp).any():
        print("Rejected " + sample['jid'] + ":\t Dielectric function values NaN")
        reject_count += 1
        continue

    if np.all(interp==0):
        print("Rejected " + sample['jid'] + ":\t Dielectric function values zero")
        reject_count += 1
        continue
    
    rejected = False
    for i in range(0, len(interp)):
        if i * 0.05 < float(sample['mbj_bandgap']) and interp[i] > 5.0:
            rejected = True
    
    if rejected:
        print("Rejected " + sample['jid'] + ":\t Invalid behavior within bandgap")
        reject_count += 1
        continue

    
    try:
        atoms = Atoms.from_dict(sample['atoms'])
    except KeyError:
        print("Rejected " + sample['jid'] + ":\t Could not construct atoms")
        reject_count += 1
        continue
    name = 'POSCAR-' + sample['jid'] + '.vasp'
    names.append(name)
    atoms.write_poscar('structure_data/' + name)
    lines.append(interp)
    count += 1
    if count == SAMPLE_COUNT:
        break

    # Progress info:
    progress = (count / SAMPLE_COUNT) * 100
    if progress >= next_threshold:
        print("Parsing and collecting samples:\t " + str(next_threshold) + "%")
        next_threshold += threshold_offset

print("Valid samples processed: " + str(count) + "/" + str(count + reject_count))
print("Rejected samples: " + str(reject_count) + "/" + str(count + reject_count))

print("Detected maximum value is " + str(max_value))
max_value = 1000.0

f = open('output_data/norm_div.txt', 'w')
f.write(str(max_value))
f.close()

print("Writing id_prop...")

f = open('structure_data/id_prop.csv', 'w')

for i in range(0, len(lines)):
    f.write(names[i]+','+','.join(map(str, map(lambda x: x / max_value, lines[i]))) + '\n')

f.close()

print("id_prop done")