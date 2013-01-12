# solve_core_recovery.py

# Imports
import os
from util.cmdline import *
from time import time
from misc import pickle_
from functools import partial
import multiprocessing as mp

# Constants
FINAL_SOLVER_OPTIONS = dict(maxIterations=100, 
                            gradientThreshold=1e-6,
                            updateThreshold=1e-7,
                            improvementThreshold=1e-8,
                            verbosenessLevel=1)

# parse_args
def parse_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument('action', choices=['update_silhouette',
                                           'solve_continuous'])
    parser.add_argument('solver')
    parser.add_argument('working')
    parser.add_argument('outer_loops', type=str, default='range(20)')
    parser.add_argument('--lambdas', type=str, default='None')
    parser.add_argument('--preconditioners', type=str, default='None')
    parser.add_argument('--max_restarts', type=int, default=10)
    parser.add_argument('--candidate_radius', type=float, default=None)
    parser.add_argument('--solver_options', type=str, default='{}')
    parser.add_argument('--num_processes', type=int, default=mp.cpu_count())

    args = parser.parse_args()

    for key in ['outer_loops', 
                'solver_options',
                'lambdas',
                'preconditioners']:
        setattr(args, key, eval(getattr(args, key)))

    for key, dtype in [
        ('lambdas', np.float64),
        ('preconditioners', np.float64)]:
        a = getattr(args, key)
        if a is not None:
            setattr(args, key, np.asarray(a, dtype=dtype))

    if not args.outer_loops:
        raise ValueError('empty "outer_loops"')

    solver_options = FINAL_SOLVER_OPTIONS.copy()
    solver_options.update(args.solver_options)
    args.solver_options = solver_options

    if '{default}' in args.working:
        make_str = lambda a: ",".join('%.4g' % a_ for a_ in a)
        mesh_file = os.path.split(args.mesh)[1]
        arap_file = os.path.split(args.arap_sections)[1]

        if len(args.indices) > 10:
            indices_str = (make_str(args.indices[:5]) + '---' +
                           make_str(args.indices[-5:]))
        else:
            indices_str = make_str(args.indices)

        default_directory = '%s_%s_%s_%s_%s' % (
            os.path.splitext(mesh_file)[0],
            os.path.splitext(arap_file)[0],
            indices_str,
            make_str(args.lambdas), 
            make_str(args.preconditioners))

        args.working = args.working.format(default=default_directory)

    return args

# Timer
class Timer(object):
    def __init__(self):
        self._t0 = time()

    def __call__(self):
        return time() - self._t0

# save_states
def save_states(working, iteration, index, states, verbose=True):
    path = os.path.join(working, str(iteration))
    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, str(index) + '.dat')

    if verbose:
        print '-> %s' % full_path

    pickle_.dump(full_path, states)

# main
def main():
    args = parse_args()

    solver = pickle_.load(args.solver)

    to_swap = dict(solver_options=args.solver_options,
                   max_restarts=args.max_restarts,
                   lambdas=args.lambdas,
                   preconditioners=args.preconditioners)
                  
    def swap_solver_options():
        for key, arr in to_swap.iteritems():
            if arr is None:
                continue

            to_swap[key] = getattr(solver, key)
            setattr(solver, key, arr)
    swap_solver_options()

    save_solver_states = partial(save_states, args.working, verbose=True)
    overall_time = Timer()

    if args.action == 'update_silhouette':
        solver.parallel_solve_silhouettes(n=args.num_processes,
                                          chunksize=solver.n /
                                          args.num_processes,
                                          verbose=True)
    else:
        # first forward-pass with no silhouette used
        if args.outer_loops[0] == -1:
            print '[# -1]'
            for i in xrange(solver.n):
                if i > 0:
                    solver._s.V1[i].flat = solver._s.V1[i-1].flat

                callback, states = solver.solve_instance_callback(i)

                solver.solve_instance(i, fixed_global_rotation=False,
                               fixed_scale=True,
                               no_silhouette=True,
                               callback=callback)

                solver.solve_instance(i, fixed_global_rotation=False,
                               fixed_scale=False,
                               no_silhouette=True,
                               callback=callback)

                save_solver_states(-1, i, states)

            args.outer_loops.pop(0)

        for l in args.outer_loops:
            print '[# %d] Complete:' % l
            outer_timer = Timer()

            # solve for the core geometry
            callback, core_states = solver.solve_core_callback()

            print '[# %d] Core:' % l
            t = solver.solve_core(callback=callback)
            print '[# %d] Core: %.3fs' % (l, t)

            save_solver_states(l, 'core', core_states)

            # solve for each instance
            for i in xrange(solver.n):
                callback, states = solver.solve_instance_callback(i)

                print '[# %d] Instance: %d' % (l, i)
                t = solver.solve_instance(
                    i, fixed_global_rotation=False,
                    fixed_scale=False,
                    no_silhouette=False,
                    callback=callback)
                print '[# %d] Instance: %d, %.3fs' % (l, i, t)
                save_solver_states(l, i, states)

            print '[# %d] Complete: %.3fs' % (l, outer_timer())

    print 'Complete time taken: %.3fs' % overall_time()

    swap_solver_options()

    head, tail = os.path.split(args.solver)
    root, ext = os.path.splitext(tail)

    split = root.split('_')
    try:
        n = int(split[-1]) + 1
    except ValueError:
        n = 0

    output_path = os.path.join(head, '_'.join(split[:-1]) + '_%d.dat' % n)
    print '-> %s' % output_path
    pickle_.dump(output_path, solver)

    args_path = os.path.join(head, root + '_ARGS.dat')
    print '-> %s' % args_path
    pickle_.dump(args_path, args.__dict__)
    
if __name__ == '__main__':
    main()

