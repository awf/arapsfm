# mp.py

# Imports
import multiprocessing as mp
from pprint import pprint
from itertools import izip_longest

# grouper (from `itertools`)
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

# async_exec
def async_exec(f, iterable, n=None, timeout=5., chunksize=1, verbose=False):
    def _async_exec_process(args_list):
        for args in args_list:
            if args is None:
                return

            f(*args)

    if n is None:
        n = mp.cpu_count()

    if n == 1:
        return _async_exec_process(iterable)

    active_processes = []

    def wait_for_active_processes(n):
        while len(active_processes) >= n:
            if verbose:
                print 'active_processes:'
                pprint(map(lambda p: p.pid, active_processes))
            for i, p in enumerate(active_processes):
                if verbose:
                    print 'join: %d' % p.pid
                p.join(timeout)

                if not p.is_alive():
                    break
            else:
                continue

            if verbose:
                print 'deleting: %d' % p.pid
            del active_processes[i]

    for args_list in grouper(chunksize, iterable, None):
        wait_for_active_processes(n)
        
        # launch next process
        p = mp.Process(target=_async_exec_process, args=(args_list, ))
        p.start()
        active_processes.append(p)

    wait_for_active_processes(1)

