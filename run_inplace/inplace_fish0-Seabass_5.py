# inplace_fish0-Seabass_5.py

# Imports
import argparse
import os, sys
from multiprocessing import Pool, cpu_count
from pprint import pprint

# SINGLE_ARAP_CMD
SINGLE_ARAP_CMD = '''python solve_single.py "single_arap_proj" \
data/models/Seabass_5.npz \
data/user_constraints/fish0/Seabass_5/{i:d}.npz --use_linear_transform \
"1.0,1.0" \
--solver_options "dict(maxIterations=200, improvementThreshold=1e-6)" \
--input_frame data/frames/fish0/{i:d}.png \
--output Fish0-Seabass_5/ARAP_Proj/{i:d}.dat \
> Fish0-Seabass_5/ARAP_Proj/{i:d}.log'''

# SINGLE_SILHOUETTE_CMD
SINGLE_SILHOUETTE_CMD = '''python generate_silhouette_info.py \
--step 40. \
data/models/Seabass_5.npz \
data/user_constraints/fish0/Seabass_5/{i:d}.npz --use_linear_transform \
Fish0-Seabass_5/Silhouette_Info/{i:d}.npz \
> Fish0-Seabass_5/Silhouette_Info/{i:d}.log'''

# CMD_FORMAT_LOOKUP
CMD_FORMAT_LOOKUP = {'arap' : SINGLE_ARAP_CMD,
                     'silhouette' : SINGLE_SILHOUETTE_CMD}

# system
def system(cmd):
    print '[%d]:' % os.getpid(), cmd
    os.system(cmd)

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('solver', choices=CMD_FORMAT_LOOKUP.keys(), type=str)
    parser.add_argument('indices', type=str)
    args = parser.parse_args()
    for key in ['indices']:
        setattr(args, key, eval(getattr(args, key)))

    pprint(args)

    all_cmds = map(lambda i: CMD_FORMAT_LOOKUP[args.solver].format(i=i), 
                   args.indices)

    p = Pool(cpu_count())
    p.map(system, all_cmds)

if __name__ == '__main__':
    main()        
    
