# visualise_standalone.py

# Imports
import os
import inspect
import argparse
from visualise import *
from itertools import ifilter, count
from functools import wraps

# requires
def requires(*keys, **kwargs):
    keys = list(keys)
    attrs = kwargs.get('attrs', [])
    def_return = lambda *args, **kwargs: None

    def fn_wrapper(fn):
        @wraps(fn)
        def wrapped_fn(self, *args, **kwargs):
            for attr in attrs:
                if not hasattr(self, attr):
                    return def_return
                keys.append(getattr(self, attr))

            for key in keys:
                if not key in self.z:
                    return def_return

            return fn(self, *args, **kwargs)

        return wrapped_fn

    return fn_wrapper
        
# StandaloneVisualisation
class StandaloneVisualisation(object):
    def __init__(self, filename, **kwargs):
        restrict_setup = kwargs.pop('restrict_setup', None)

        self.z = np.load(filename)
        self.vis = VisualiseMesh()

        self.used = set()
        self.__dict__.update(kwargs)

        self.setup_processed = set()
        self.setup(restrict_setup)

    def setup(self, restrict_setup=None):
        add_methods = ifilter(lambda t: t[0].startswith('_add_'),
                              inspect.getmembers(self))

        for method_name, bound_method in add_methods:
            if (restrict_setup is not None and 
                method_name not in restrict_setup):
                continue

            if method_name in self.setup_processed:
                continue

            bound_method()
            self.setup_processed.add(method_name)

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
    @requires('T', attrs=['vertices_key'])
    def _add_mesh(self):
        self.vis.add_mesh(self[self.vertices_key], 
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

    @requires('C', 'P')
    def _add_projection(self):
        C = self['C']
        P = self['P']
        self.vis.add_projection(C, P)

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

# bool_from_string
def bool_from_string(string):
    if string == 'True': return True
    if string == 'False' : return False

    raise ValueError('unable to create bool from string: %s' %
                     string)

# tuple_from_string
def tuple_from_string(string):
    if not string: return tuple()

    tup_strings = string.split(',')
    for type_ in [int, float, bool_from_string]:
        try:
            return tuple(type_(v) for v in tup_strings)
        except ValueError:
            continue

    raise ValueError('unable to create tuple from string: %s' %
                     string)

# parse_camera_action
def parse_camera_action(action):
    # get method and value string
    method, value_string = action.split('=')

    # check if immediately save rendering after this camera action
    save_after = True
    if value_string.endswith(','):
        save_after = False
        value_string = value_string[:-1]
    
    # convert value string to tuple
    tup = tuple_from_string(value_string)

    return method, tup, save_after 

# parse_actor_properties
def parse_actor_properties(action):
    name, remainder = action.split(':')
    method, value_string = remainder.split('=')
    tup = tuple_from_string(value_string)

    return name, method, tup 

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
    parser.add_argument('-a', dest='actor_properties', type=str, default=[],
                        action='append', help='actor properties')
    parser.add_argument('--magnification', type=int, default=1,
                        help='magnification')
    parser.add_argument('--vertices-key', type=str, default='V',
                        help='key to retrieve vertices')
    parser.add_argument('--output_image_first', 
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    # echo arguments
    print 'Arguments:', args

    # setup visualisation
    print 'Source file: %s' % args.input
    print 'Available keys:', np.load(args.input).keys()

    restrict_setup = ('_add_image,') if args.output_image_first else set([])

    vis = StandaloneVisualisation(args.input,
                                  vertices_key=args.vertices_key,
                                  restrict_setup=restrict_setup)

    # is visualisation interface or to file?
    interactive_session = args.output_directory is None

    # setup output directory
    if not interactive_session and not os.path.exists(args.output_directory):
        print 'Creating directory: ', args.output_directory
        os.makedirs(args.output_directory)

    # peform camera actions and save outputs as required
    n = count(0)

    # if output image first then save now
    if args.output_image_first:
        if vis.actors:
            full_path = os.path.join(args.output_directory, '%d.png' % next(n))
            print 'Output: ', full_path
            vis.write(full_path, magnification=args.magnification)

        vis.setup()

    # process actor properties
    for action in args.actor_properties:
        actor_name, method, tup = parse_actor_properties(action)
        vis.actor_properties(actor_name, (method, tup))

    for action in args.camera_actions:
        # parse the action
        method, tup, save_after = parse_camera_action(action)
        print '%s(*%s), save_after=%s' % (method, tup, save_after)

        # execute the camera action
        vis.camera_actions((method, tup))

        # save if required
        if not interactive_session and save_after:
            full_path = os.path.join(args.output_directory, '%d.png' % next(n))
            print 'Output: ', full_path
            vis.write(full_path, magnification=args.magnification)

    # show if interactive
    if interactive_session:
        print 'Interactive'
        vis.execute(magnification=args.magnification)

# test_tuple_from_string
def test_tuple_from_string():
    print tuple_from_string('1')

if __name__ == '__main__':
    main()
    # test_tuple_from_string()
    
