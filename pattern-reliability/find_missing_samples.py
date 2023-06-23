import os


root_dir = '/media/tuttj/phd_data/datasets/pattern_reliability_tifs_2023_hist_matched'

# Look for originals

for run in [1]:

    removed = set()

    orig = os.path.join('scanner', f'run_{run}', 'original')
    fake = os.path.join('scanner', f'run_{run}', 'fake')

    if run == 0:
        M = 720
    else:
        M = 1440

    for file1 in [orig, fake]:
        for N in range(1,M+1):
            path = os.path.join(file1, f'{N}'.zfill(6) + '.tiff')
            abs_path = os.path.join(root_dir, path)
            if not os.path.exists(abs_path):
                removed.add(N)

    removed = list(removed)
    removed.sort()
    removed = ','.join(str(i) for i in removed)


    with open(os.path.join(root_dir, 'removed_scanner.csv'), 'a') as f:
        f.write(removed + '\n')


"""
root_dir = '/media/tuttj/phd_data/datasets/2023_Indigo_1x1_mobile'

with open(os.path.join(root_dir, 'excluded_roman.txt'), 'r') as f:
    excluded = f.read()

# Look for originals

orig = os.path.join('orig_phone', 'HPI55_printdpi812.8_printrun1_session1_InvercoteG')
fake = os.path.join('fake_phone', 'HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55')

for phone in ['iPhone12Pro', 'SamsungGN20U']:
    for run in range(1,7):
        removed = ''
        for file1 in [orig, fake]:
            file2 = os.path.join(file1, f'{phone}_run{run}_ss100_focal12_apperture1', 'rcod')
            for N in range(1,1441):
                path = os.path.join(file2, f'{N}'.zfill(6) + '.tiff')
                abs_path = os.path.join(root_dir, path)
                if not os.path.exists(abs_path):
                    removed += f'{N},'
                elif path in excluded:
                    removed += f'{N},'

        if phone == 'iPhone12Pro':
            with open(os.path.join(root_dir, 'removed_iphone.csv'), 'a') as f:
                f.write(removed + '\n')
        if phone == 'SamsungGN20U':
            with open(os.path.join(root_dir, 'removed_samsung.csv'), 'a') as f:
                f.write(removed + '\n')

"""