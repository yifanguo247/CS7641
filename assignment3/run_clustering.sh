#!/bin/sh

# Replace 'X' below with the optimal values found
# If you want to first generate data and updated datasets, remove the "--skiprerun" flags below

# python run_experiment.py --ica --bank --dim 7  --skiprerun --verbose --threads -1 > ica-bank-clustering.log 2>&1
# python run_experiment.py --ica --htru2   --dim X  --skiprerun --verbose --threads -1 > ica-htru2-clustering.log   2>&1
# python run_experiment.py --pca --bank --dim 3  --skiprerun --verbose --threads -1 > pca-bank-clustering.log 2>&1
# python run_experiment.py --pca --htru2   --dim X  --skiprerun --verbose --threads -1 > pca-htru2-clustering.log   2>&1
# python run_experiment.py --rp  --bank --dim 10  --skiprerun --verbose --threads -1 > rp-bank-clustering.log  2>&1
# python run_experiment.py --rp  --htru2   --dim X  --skiprerun --verbose --threads -1 > rp-htru2-clustering.log    2>&1
# python run_experiment.py --rf  --bank --dim 1  --skiprerun --verbose --threads -1 > rf-bank-clustering.log  2>&1
# python run_experiment.py --rf  --htru2   --dim X  --skiprerun --verbose --threads -1 > rf-htru2-clustering.log    2>&1
# python run_experiment.py --svd --statlog --dim X  --skiprerun --verbose --threads -1 > svd-statlog-clustering.log 2>&1
# python run_experiment.py --svd --htru2   --dim X  --skiprerun --verbose --threads -1 > svd-htru2-clustering.log   2>&1



python run_experiment.py --ica --mnist --dim 16  --skiprerun --verbose --threads -1 > ica-mnist-clustering.log 2>&1
python run_experiment.py --pca --mnist --dim 21  --skiprerun --verbose --threads -1 > pca-mnist-clustering.log 2>&1
python run_experiment.py --rp  --mnist --dim 9  --skiprerun --verbose --threads -1 > rp-mnist-clustering.log  2>&1
python run_experiment.py --rf  --mnist --dim 65  --skiprerun --verbose --threads -1 > rf-mnist-clustering.log  2>&1