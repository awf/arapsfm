# inplace_cheetah1_Cheetah4.py

# Imports
import argparse
import numpy as np
import os
from os.path import join, expanduser
from fbg_cluster.job_submission import process_jobs
from pprint import pprint
from itertools import combinations
import subprocess

# Logging
import logging
logging.basicConfig(format='<%(asctime)s> %(levelname)s.%(process)d.%(funcName)s #  %(message)s', level=logging.DEBUG)

# number_string
def number_string(floats):
    return '"' + ','.join('%s' % float_ for float_ in floats) + '"' 

# make_args
def make_args(indices, lambdas, preconditioners, solver_options,
              max_restarts, output_dir):

    args = ['data/models/Cheetah_4.npz',    
            'data/user_constraints/cheetah1/Cheetah_4_4.npz',
            '--use_linear_transform',
            '"(' + number_string(indices)[1:-1] + ')"',
            'cheetah1/Cheetah_4_%d_0.dat',
            'data/user_constraints/cheetah1/Cheetah_4_%d.npz',
            'data/silhouettes/cheetah1/%d_S.npz',
            'cheetah1/Cheetah_4_%d_Silhouette_Info.dat',
            'data/distance_maps/cheetah1/%d_D.npz',
            number_string(lambdas),
            number_string(preconditioners),
            '--solver_options', '"' + repr(solver_options) + '"',
            '--max_restarts', str(max_restarts),
            '--uniform_weights',
            '--frames', 'data/frames/cheetah1/%d.png',
            '--output', output_dir]

    return args

PROJECT_ROOT = expanduser('~/Code/Projects/Core_Recovery')
EXPERIMENT_ROOT = join(PROJECT_ROOT, 'cheetah1_Cheetah4')

# Tests

# test_laplacian_lambdas
def test_laplacian_lambdas(number_of_queues, per_node):
    # setup test directories
    experiment_root = join(EXPERIMENT_ROOT, 'test_laplacian_lambdas')
    log_dir = join(experiment_root, 'logs')
    output_data = join(experiment_root, 'output_data')

    for dir_ in (log_dir, output_data):  
        if not os.path.exists(dir_):
            print 'Creating:', dir_
            os.makedirs(dir_)

    all_indices = [(1, 2), (3, 4)]

    base_lambdas = np.r_[1e0, 1e0, 1e3,  # silhouette
                         1e0,            # as-rigid-as-possible 
                         1e0,            # spillage
                         5e0]            # laplacian

    laplacian_lambdas = [1e0, 2e0, 3e0, 4e0, 5e0]

    all_args = []
    for indices in all_indices:
        for lambda_ in laplacian_lambdas:
            # set output directory
            output_dir = join(output_data, '%d' % len(all_args))
            if not os.path.exists(output_dir):
                print 'Creating:', output_dir
                os.makedirs(output_dir)

            # set Laplacian lambda
            lambdas = base_lambdas.copy()
            lambdas[-1] = lambda_

            all_args.append(
                make_args(indices,
                          lambdas,
                          (1.0,1.0,5.0), 
                          dict(improvementThreshold=1e-4),
                          10,
                          output_dir))

    print '# of tests:', len(all_args)

    process_jobs('test_laplacian_lambdas',  
                 PROJECT_ROOT,
                 'solve_multiple.py',
                 all_args,
                 number_of_queues,
                 per_node,
                 submit_args=['-o', log_dir, '-e', log_dir])

# test_frame_subsets
def test_frame_subsets(number_of_queues, per_node):
    # setup test directories
    experiment_root = join(EXPERIMENT_ROOT, 'test_frame_subsets')
    log_dir = join(experiment_root, 'logs')
    output_data = join(experiment_root, 'output_data')

    for dir_ in (log_dir, output_data):  
        if not os.path.exists(dir_):
            print 'Creating:', dir_
            os.makedirs(dir_)

    lambdas = np.r_[1e0, 1e0, 1e3,  # silhouette
                    1e0,            # as-rigid-as-possible 
                    1e0,            # spillage
                    2e0]            # laplacian

    indices = [1, 2, 3, 4]

    all_indices = [] 
    for i in (2, 3, 4):
        all_indices += list(combinations(indices, i))
        
    all_args = []
    for indices in all_indices:
        # set output directory
        output_dir = join(output_data, '%d' % len(all_args))
        if not os.path.exists(output_dir):
            print 'Creating:', output_dir
            os.makedirs(output_dir)

        all_args.append(
            make_args(indices,
                      lambdas,
                      (1.0,1.0,5.0), 
                      dict(improvementThreshold=1e-4),
                      10,
                      output_dir))

    print '# of tests:', len(all_args)

    process_jobs('test_frame_subsets',  
                 PROJECT_ROOT,
                 'solve_multiple.py',
                 all_args,
                 number_of_queues,
                 per_node,
                 submit_args=['-o', log_dir, '-e', log_dir])

# test_arap_lambda
def test_arap_lambda(number_of_queues, per_node):
    # setup test directories
    experiment_root = join(EXPERIMENT_ROOT, 'test_arap_lambda')
    log_dir = join(experiment_root, 'logs')
    output_data = join(experiment_root, 'output_data')

    for dir_ in (log_dir, output_data):  
        if not os.path.exists(dir_):
            print 'Creating:', dir_
            os.makedirs(dir_)

    base_lambdas = np.r_[1e0, 1e0, 1e3,  # silhouette
                         1e0,            # as-rigid-as-possible 
                         1e0,            # spillage
                         2e0]            # laplacian

    indices = [1, 2, 3, 4]

    all_args = []
    for lambda_ in (1e0, 2e0, 5e0, 1e1, 1e2):
        # set output directory
        output_dir = join(output_data, '%d' % len(all_args))
        if not os.path.exists(output_dir):
            print 'Creating:', output_dir
            os.makedirs(output_dir)

        # set arap lambda
        lambdas = base_lambdas.copy()
        lambdas[3] = lambda_
        
        all_args.append(
            make_args(indices,
                      lambdas,
                      (1.0,1.0,5.0), 
                      dict(improvementThreshold=1e-4),
                      10,
                      output_dir))

    print '# of tests:', len(all_args)

    process_jobs('test_arap_lambda',  
                 PROJECT_ROOT,
                 'solve_multiple.py',
                 all_args,
                 number_of_queues,
                 per_node,
                 submit_args=['-o', log_dir, '-e', log_dir])

# test_laplacian_lambda2
def test_laplacian_lambda2(number_of_queues, per_node):
    # setup test directories
    experiment_root = join(EXPERIMENT_ROOT, 'test_laplacian_lambda2')
    log_dir = join(experiment_root, 'logs')
    output_data = join(experiment_root, 'output_data')

    for dir_ in (log_dir, output_data):  
        if not os.path.exists(dir_):
            print 'Creating:', dir_
            os.makedirs(dir_)

    base_lambdas = np.r_[1e0, 1e0, 1e3,  # silhouette
                         1e0,            # as-rigid-as-possible 
                         1e0,            # spillage
                         2e0]            # laplacian

    indices = [1, 2, 3, 4]

    all_args = []
    for lambda_ in (1e0, 2e0, 5e0, 1e1):
        # set output directory
        output_dir = join(output_data, '%d' % len(all_args))
        if not os.path.exists(output_dir):
            print 'Creating:', output_dir
            os.makedirs(output_dir)

        # set Laplacian lambda
        lambdas = base_lambdas.copy()
        lambdas[5] = lambda_
        
        all_args.append(
            make_args(indices,
                      lambdas,
                      (1.0,1.0,5.0), 
                      dict(improvementThreshold=1e-4),
                      10,
                      output_dir))

    print '# of tests:', len(all_args)

    print 'last arguments:'
    print ' '.join(all_args[-1])
    raw_input()

    process_jobs('test_laplacian_lambda2',  
                 PROJECT_ROOT,
                 'solve_multiple.py',
                 all_args,
                 number_of_queues,
                 per_node,
                 submit_args=['-o', log_dir, '-e', log_dir])

# test_laplacian_lambda3
def test_laplacian_lambda3(number_of_queues, per_node):
    # setup test directories
    experiment_root = join(EXPERIMENT_ROOT, 'test_laplacian_lambda3')
    log_dir = join(experiment_root, 'logs')
    output_data = join(experiment_root, 'output_data')

    for dir_ in (log_dir, output_data):  
        if not os.path.exists(dir_):
            print 'Creating:', dir_
            os.makedirs(dir_)

    base_lambdas = np.r_[1e0, 1e0, 1e3,  # silhouette
                         1e0,            # as-rigid-as-possible 
                         1e0,            # spillage
                         2e0]            # laplacian

    all_indices = list(combinations([1, 2, 3, 4], 3))
    all_args = []

    for indices in all_indices:
        for lambda_ in (1e0, 2e0, 5e0, 1e1):
            # set output directory
            output_dir = join(output_data, '%d' % len(all_args))
            if not os.path.exists(output_dir):
                print 'Creating:', output_dir
                os.makedirs(output_dir)

            # set Laplacian lambda
            lambdas = base_lambdas.copy()
            lambdas[5] = lambda_
            
            all_args.append(
                make_args(indices,
                          lambdas,
                          (1.0,1.0,5.0), 
                          dict(improvementThreshold=1e-4),
                          10,
                          output_dir))

    print '# of tests:', len(all_args)

    process_jobs('test_laplacian_lambda3',  
                 PROJECT_ROOT,
                 'solve_multiple.py',
                 all_args,
                 number_of_queues,
                 per_node,
                 submit_args=['-o', log_dir, '-e', log_dir])

# main_inplace
def main_inplace():
    args = make_args(
           (1,2), 
           (1e0,1e0,1e3,1e0,1e0,2e0),
           (1.0,1.0,5.0), 
           dict(improvementThreshold=1e-4),
           10,
           'cheetah1')

    pprint(args)

    subprocess.check_call(['python', 'solve_multiple.py'] + args)

# main
def main():
    parser = argparse.ArgumentParser(
        description='cheetah1_Cheetah4_multiple_solve Tests')
    parser.add_argument('number_of_queues', type=int)
    parser.add_argument('per_node', type=int)

    cmdline_args = parser.parse_args()

    # test_laplacian_lambdas(cmdline_args.number_of_queues,
    #                        cmdline_args.per_node)
    # test_frame_subsets(cmdline_args.number_of_queues,
    #                    cmdline_args.per_node)
    # test_arap_lambda(cmdline_args.number_of_queues,
    #                  cmdline_args.per_node)
    test_laplacian_lambda2(cmdline_args.number_of_queues,
                           cmdline_args.per_node)
    # test_laplacian_lambda3(cmdline_args.number_of_queues,
    #                        cmdline_args.per_node)

if __name__ == '__main__':
    main()

