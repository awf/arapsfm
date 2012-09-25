# visualise_standalone.py

# Imports
import os
import inspect
import argparse
from visualise import *
from itertools import ifilter, count
from functools import wraps

# requires
def requires(*keys):
    def fn_wrapper(fn):
        @wraps(fn)
        def wrapped_fn(self, *args, **kwargs):
            for key in keys:
                if not key in self.z:
                    #raise KeyError('%s requires "%s"' % (fn.func_name, key))
                    return lambda *args, **kwargs: None

            return fn(self, *args, **kwargs)

        return wrapped_fn

    return fn_wrapper
        
# StandaloneVisualisation
class StandaloneVisualisation(object):
    def __init__(self, filename):
        self.z = np.load(filename)
        self.vis = VisualiseMesh()
        self.used = set()
        self.setup()

    def setup(self):
        add_methods = ifilter(lambda t: t[0].startswith('_add_'),
                              inspect.getmembers(self))

        for method_name, bound_method in add_methods:
            bound_method()

    # safe lookup from "z"
    def __getitem__(self, key):
        try:
            r = self.z[key]
            self.used.add(key)

        except KeyError:
            r = None

        return r

    def __getattr__(self, attr):
        if attr.startswith('__'):
            return None

        return getattr(self.vis, attr)

    # "_add" methods
    @requires('V', 'T')
    def _add_mesh(self):
        self.vis.add_mesh(self['V'], 
                          self['T'],
                          self['L'])

    @requires('image')
    def _add_image(self):
        self.vis.add_image(str(self['image']))

    @requires('Q', 'S')
    def _add_silhouette(self):
        Q = self['Q']
        S = self['S']
        N = Q.shape[0]
        self.vis.add_silhouette(Q, np.arange(N), [0, N-1], S)

# TestClass
class TestClass(object):
    def __init__(self):
        self.z = {'a' : 1}
        test_members = filter(lambda t: t[0].startswith('test'),
                              inspect.getmembers(self))
        print test_members

    @requires('a')
    def test1(self):
        print self.z['a']

    @requires('b')
    def test2(self):
        print self.z['b']

# test_TestClass
def test_TestClass():
    c = TestClass()
    c.test1()
    c.test2()

# parse_camera_action
def parse_camera_action(action):
    # get method and value string
    method, value_string = action.split('=')

    # check if immediately save rendering after this camera action
    save_after = True
    if value_string.endswith(','):
        save_after = False
        value_string = value_string[:-1]

    # convert value string to value
    for type_ in [int, float, bool]:
        try:
            value = type_(value_string)
            break
        except ValueError:
            continue
    else:
        raise ValueError('unable to parse value string: %s' %
                         value_string)

    return method, value, save_after 

# main
def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser(
        description='Visualise deformation results')
    parser.add_argument('input', type=str, help='input deformations file')
    parser.add_argument('-o', dest='output_directory', type=str, default=None,
                        help='output directory')
    parser.add_argument('-c', dest='camera_actions', type=str, default=[],
                        action='append', help='camera actions')
    parser.add_argument('--magnification', type=int, default=1,
                        help='magnification')

    args = parser.parse_args()

    # setup visualisation
    print 'Source file: %s' % args.input
    vis = StandaloneVisualisation(args.input)

    # is visualisation interface or to file?
    interactive_session = args.output_directory is None

    # setup output directory
    if not interactive_session and not os.path.exists(args.output_directory):
        print 'Creating directory: ', args.output_directory
        os.makedirs(args.output_directory)

    # peform camera actions and save outputs as required
    n = count(0)

    for action in args.camera_actions:
        # parse the action
        method, value, save_after = parse_camera_action(action)
        print '%s(%s), save_after=%s' % (method, value, save_after)

        # execute the camera action
        vis.camera_actions((method, value))

        # save if required
        if not interactive_session and save_after:
            full_path = os.path.join(args.output_directory, '%d.png' % next(n))
            print 'Output: ', full_path
            vis.write(full_path, magnification=args.magnification)

    # show if interactive
    if interactive_session:
        print 'Interactive'
        vis.execute(magnification=args.magnification)

if __name__ == '__main__':
    main()
    
