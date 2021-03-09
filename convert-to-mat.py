import os
import sys
import re
import numpy as np
from scipy.io import savemat
from progress import ProgressBar

CATEGORIES = ["02691156", "02773838", "02954340", "02958343", "03001627",
              "03261776", "03467517", "03624134", "03636649", "03642806",
              "03790512", "03797390", "03948459", "04099429", "04225987", "04379243"]

CATEGORY_NAMES = ['airplane', 'bag', 'cap', 'car', 'chair',
                 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']

USAGE = """\
Usage:  convert-to-mat point-files-directory seg-directory mat-dir

    point-files-directory : where to look for point-files
    seg-directory : where to look for seg-files
    mat-dir : where to save the mat-files
"""

if len(sys.argv) != 4:
    print(USAGE)
    exit(0)


POINTS_DIRECTORY = sys.argv[1]
SEG_DIRECTORY = sys.argv[2]
MATDIR = sys.argv[3]


print(f"Points: {POINTS_DIRECTORY}\nLabels: {SEG_DIRECTORY}\nMat dir: {MATDIR}")
if input("Coninue? (yes/no) ").strip() != "yes": exit(0)


print("Avaliable categories:")
print(' '.join(CATEGORY_NAMES))
chosen_categories = sorted(input("Which categories to use? (write names separated by space) ").strip().lower().split(" "))
CATEGORIES = [CATEGORIES[CATEGORY_NAMES.index(c)] for c in chosen_categories]
CATEGORY_NAMES = chosen_categories


def read_csv(path):
    with open(path) as f:
         data = f.read()
    for row in data.strip().split('\n'):
        yield tuple(map(float, map(str.strip, re.split(r'[ ,;]', row))))


def make_mat(pc_id, category):
    points = tuple(read_csv(os.path.join(POINTS_DIRECTORY, category, f'{pc_id}.pts')))
    seg = tuple(read_csv(os.path.join(SEG_DIRECTORY, category, f'{pc_id}.seg')))
    point_array = np.array(points)
    label_array = np.array(seg)
    category_array = np.ones((1, 1)) * CATEGORIES.index(category)
    savemat(os.path.join(MATDIR, f'{pc_id}.mat'),
            {'points': point_array, 'labels': label_array, 'category': category_array})


num_files = sum(map(len, (os.listdir(os.path.join(POINTS_DIRECTORY, category))
                          for category in CATEGORIES)))


print(f"Creating {num_files} files")
progress_step = max(num_files // 1000, 1)
progress_bar = ProgressBar(num_files)
i = 0

for category in CATEGORIES:
    for pointfile in os.listdir(os.path.join(POINTS_DIRECTORY, category)):
        pc_id = pointfile[:-4]
        if i % progress_step == 0:
            progress_bar.update(i)

        make_mat(pc_id, category)
        i += 1
print('')
