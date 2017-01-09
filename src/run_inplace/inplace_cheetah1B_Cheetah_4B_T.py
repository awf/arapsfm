# inplace_cheetah1B_Cheetah_4B_T.py

# Imports
import argparse
from os import system

# SINGLE_ARAP_CMD
SINGLE_ARAP_CMD = '''python solve_single.py "single_arap_proj" \
data/models/Cheetah_4B_T.npz \
data/user_constraints/cheetah1B/Cheetah_4B_T/{frame_index}.npz \
--use_linear_transform "1.0,1.0" \
--solver_options "dict(maxIterations=200, improvementThreshold=1e-6)" \
--input_frame data/frames/cheetah1B/{frame_index}.png \
--output cheetah1B/Cheetah_4B_T/Cheetah_4B_T_{frame_index}_0.dat \
> cheetah1B/Cheetah_4B_T/Cheetah_4B_T_{frame_index}_0.log'''

# main_single_arap
def main_single_arap():
    # for all frame indices
    for i in xrange(16):
        cmd = SINGLE_ARAP_CMD.format(frame_index=i)
        print '-> %s' % cmd
        system(cmd)

# SINGLE_SILHOUETTE_CMD
SINGLE_SILHOUETTE_CMD = '''python generate_silhouette_info.py \
--step 30. \
data/models/Cheetah_4B_T.npz \
data/user_constraints/cheetah1B/Cheetah_4B_T/{frame_index}.npz \
--use_linear_transform \
cheetah1B/Cheetah_4B_T/Cheetah_4B_T_{frame_index}_Silhouette_Info.dat \
> cheetah1B/Cheetah_4B_T/Cheetah_4B_T_{frame_index}_Silhouette_Info.log'''

# silhouette_process
def silhouette_process(i):
    cmd = SINGLE_SILHOUETTE_CMD.format(frame_index=i)
    print '-> %s' % cmd
    system(cmd)
    
# main_single_silhouette
def main_single_silhouette():
    from multiprocessing import Pool, cpu_count
    p = Pool(cpu_count())
    p.map(silhouette_process, range(16))

MULTIVIEW_CMD = '''
'''

# main_multiview
def main_multiview():
    pass

if __name__ == '__main__':
    main_single_silhouette()
    # main_single_arap()
