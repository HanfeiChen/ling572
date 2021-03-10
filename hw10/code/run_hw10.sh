#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
# if you install anaconda in a different directory, try the following command
# source path_to_anaconda3/anaconda3/etc/profile.d/conda.sh

conda activate /dropbox/20-21/572/hw10/code/env/

cd .

#python main.py --num_epochs 6 --data_dir /dropbox/20-21/572/hw10/code/data/

# q1
python main.py --num_epochs 6 --data_dir /dropbox/20-21/572/hw10/code/data/ > q1.out

# q2
python main.py --num_epochs 6 --data_dir /dropbox/20-21/572/hw10/code/data/ --L2 > q2.out

# q3
python main.py --num_epochs 12 --data_dir /dropbox/20-21/572/hw10/code/data/ --patience 3 --L2 > q3.out

