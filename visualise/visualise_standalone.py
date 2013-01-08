# visualise_standalone.py

# Imports
import os
import inspect
import argparse
from visualise import *
from itertools import ifilter, count, izip
from functools import wraps
from operator import add, mul
from pprint import pprint

from geometry.axis_angle import *
from geometry.quaternion import *
from geometry import register
from geometry import loop
from mesh.weights import weights
from mesh.faces import faces_to_cell_array
from solvers.arap import ARAPVertexSolver

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
        with_core = kwargs.pop('with_core', None)
        self.subdivide = kwargs.pop('subdivide', 0)
        self.compute_normals = kwargs.pop('compute_normals', 0)
        self.no_colour_silhouette = kwargs.pop('no_colour_silhouette', False)

        self.z = np.load(filename)

        if with_core is not None:
            self.core_V = np.load(with_core)['V']
        else:
            self.core_V = None

        self.vis = VisualiseMesh()

        self.used = set()
        self.__dict__.update(kwargs)

        self.setup_processed = set()
        self.setup()

    def setup(self):
        add_methods = ifilter(lambda t: t[0].startswith('_add_'),
                              inspect.getmembers(self))

        for method_name, bound_method in add_methods:
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
        special_color = ((31, 120, 180) if self.no_colour_silhouette
                                        else (178, 223, 138))

        if self.subdivide == 0:
            self.vis.add_mesh(self[self.vertices_key], 
                              self['T'],
                              self['L'],
                              compute_normals=self.compute_normals,
                              special_color=special_color)
        else:
            T, V = self['T'], self[self.vertices_key]
            for i in xrange(self.subdivide):
                T, V = loop.subdivide(T, V)
                              
            self.vis.add_mesh(V, T, compute_normals=self.compute_normals,
                              special_color=special_color)

    @requires('T', 'Xg', 's', 'X', attrs=['vertices_key'])
    def _add_regular_arap(self):
        if self['y'] is not None or self.core_V is None:
            return

        T = self['T'] 
        core_V = self.core_V.copy()
        adj, W = weights(core_V, faces_to_cell_array(T), weights_type='uniform')

        solve_V = ARAPVertexSolver(adj, W, core_V)

        rotM = lambda x: rotationMatrix(quat(x))
        V1 = solve_V(map(lambda x: rotM(x), self['X']))

        xg = np.ravel(self['Xg'])
        A = self['s'] * rotM(xg)

        V1 = np.dot(V1, np.transpose(A))
        V1 += register.displacement(V1, self[self.vertices_key])

        self.vis.add_mesh(V1, T, actor_name='arap', is_base=False)

        lut = self.vis.actors['arap'].GetMapper().GetLookupTable()
        lut.SetTableValue(0, 1., 0.667, 0.)

    @requires('T', 'Xg', 's', 'X', 'y', attrs=['vertices_key'])
    def _add_basis_arap(self):
        if self.core_V is None:
            return

        X = self['X']
        y = self['y']
        if not isinstance(y, np.ndarray) or len(X) != y.shape[0]:
            return

        T = self['T'] 
        core_V = self.core_V.copy()
        adj, W = weights(core_V, faces_to_cell_array(T), weights_type='uniform')

        solve_V = ARAPVertexSolver(adj, W, core_V)

        scaled_X = []
        for y_, X_ in izip(y, X):
            scaled_X.append(map(lambda x: axScale(y_, x), X_))

        N = core_V.shape[0]
        X = []
        for i in xrange(N):
            X.append(reduce(axAdd, [X_[i] for X_ in scaled_X]))

        rotM = lambda x: rotationMatrix(quat(x))
        V1 = solve_V(map(lambda x: rotM(x), X))

        xg = np.ravel(self['Xg'])
        A = self['s'] * rotM(xg)

        V1 = np.dot(V1, np.transpose(A))

        V1 += register.displacement(V1, self[self.vertices_key])

        self.vis.add_mesh(V1, T, actor_name='arap', is_base=False)
        lut = self.vis.actors['arap'].GetMapper().GetLookupTable()
        lut.SetTableValue(0, 1., 0., 1.)

    @requires('T', 'Xg', 's', 'K', 'Xb', 'X', 'y', attrs=['vertices_key'])
    def _add_deprecated_sectioned_arap(self):
        if self.core_V is None:
            return

        # calculate `Xi`
        K = self['K']
        Xb = self['Xb']
        X = self['X']
        y = self['y']

        N = self.core_V.shape[0]
        Xi = np.zeros((N, 3), dtype=np.float64)
        for i in xrange(N):
            if K[i, 0] == 0:
                pass
            elif K[i, 0] < 0:
                Xi[i,:] = X[K[i, 1]]
            else:
                Xi[i,:] = axScale(y[K[i, 0] - 1], Xb[K[i, 1]])

        # setup `solve_V`
        T = self['T'] 
        core_V = self.core_V.copy()
        adj, W = weights(core_V, faces_to_cell_array(T), weights_type='uniform')

        solve_V = ARAPVertexSolver(adj, W, core_V)

        # solve for `V1`
        rotM = lambda x: rotationMatrix(quat(x))
        V1 = solve_V(map(lambda x: rotM(x), Xi))

        # apply global rotation and scale
        A = self['s'] * rotM(np.ravel(self['Xg']))

        V1 = np.dot(V1, np.transpose(A))

        # register by translation to the instance vertices
        V1 += register.displacement(V1, self[self.vertices_key])

        # add as orange actor
        self.vis.add_mesh(V1, T, actor_name='arap', 
                          is_base=False, 
                          color=(255, 170, 0))

    @requires('T', 'Xg', 's', 'ki', 'Xb', 'y', 'X', attrs=['vertices_key'])
    def _add_sectioned_arap(self):
        if self.core_V is None:
            return

        # calculate `Xi`
        ki = self['ki']
        Xb = self['Xb']
        y = self['y']
        X = self['X']

        N = self.core_V.shape[0]
        Xi = np.zeros((N, 3), dtype=np.float64)

        iter_ki = iter(ki)

        for i in xrange(N):
            n = next(iter_ki)
            if n == 0:
                pass
            elif n < 0:
                Xi[i, :] = X[next(iter_ki)]
            else:
                yi, Xbi = [], []
                for j in xrange(n):
                    Xbi.append(Xb[next(iter_ki)])
                    yi.append(y[next(iter_ki)])

                Xi[i, :] = reduce(add, map(mul, yi, Xbi))
                    
        # setup `solve_V`
        T = self['T'] 
        core_V = self.core_V.copy()
        adj, W = weights(core_V, faces_to_cell_array(T), weights_type='uniform')

        solve_V = ARAPVertexSolver(adj, W, core_V)

        # solve for `V1`
        rotM = lambda x: rotationMatrix(quat(x))
        V1 = solve_V(map(lambda x: rotM(x), Xi))

        # apply global rotation and scale
        A = self['s'] * rotM(np.ravel(self['Xg']))

        V1 = np.dot(V1, np.transpose(A))

        # register by translation to the instance vertices
        V1 += register.displacement(V1, self[self.vertices_key])

        # add as orange actor
        self.vis.add_mesh(V1, T, actor_name='arap', 
                          is_base=False, 
                          color=(255, 170, 0))

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
    parser.add_argument('-o', '--output', dest='output_directory', type=str, default=None,
                        help='output directory')
    parser.add_argument('-c', dest='camera_actions', type=str, default=[],
                        action='append', help='camera actions')
    parser.add_argument('--magnification', type=int, default=1,
                        help='magnification')
    parser.add_argument('--vertices-key', type=str, default='V',
                        help='key to retrieve vertices')
    parser.add_argument('--ren_win_size', type=str, default=None)
    parser.add_argument('--with_core', type=str, default=None)
    parser.add_argument('--subdivide', type=int, default=0)
    parser.add_argument('--compute_normals', action='store_true', default=False)
    parser.add_argument('--extension', type=str, default='.npz')
    parser.add_argument('--indices', type=str, default='None')
    parser.add_argument('--no_colour_silhouette',
                        action='store_true', 
                        default=False)

    args = parser.parse_args()
    args.indices = eval(args.indices)

    # echo arguments
    print 'Arguments:', args

    if os.path.isfile(args.input):
        input_paths = [args.input]
        output_paths = [args.output_directory]
    else: 
        input_files = filter(lambda f: f.endswith(args.extension),
                             os.listdir(args.input))

        def input_number(f):
            try:
                return int(os.path.splitext(f)[0].split('_')[-1])
            except ValueError:
                return None

        input_files = filter(lambda f: input_number(f) is not None,
                             input_files)

        sorted_input_files = sorted(input_files, key=input_number)

        if args.indices is not None:
            sorted_input_files = [sorted_input_files[i] for i in args.indices]
                
        input_paths = map(lambda f: os.path.join(args.input, f),
                          sorted_input_files)

        if args.output_directory is not None:
            input_numbers = map(input_number, sorted_input_files)
            output_paths = map(lambda i: os.path.join(
                               args.output_directory, str(i)), input_numbers)
        else:
            output_paths = [None] * len(input_paths)

    # is visualisation interface or to file?
    interactive_session = args.output_directory is None

    # setup output directory
    if not interactive_session:
        paths_to_make = filter(lambda p: not os.path.exists(p),
                               output_paths)
        map(os.makedirs, paths_to_make)

    vis = None

    for i, input_path in enumerate(input_paths):
        # setup visualisation
        print 'Source file: %s' % input_path
        print 'Available keys:', np.load(input_path).keys()

        if vis is None:
            vis = StandaloneVisualisation(
                input_path,
                vertices_key=args.vertices_key,
                with_core=args.with_core,
                subdivide=args.subdivide,
                compute_normals=args.compute_normals,
                no_colour_silhouette=args.no_colour_silhouette)

            if args.ren_win_size is not None:
                vis.ren_win.SetSize(*tuple_from_string(args.ren_win_size))
        else:
            # re-initialise the visualisation with-out closing the renderer
            vis.z = np.load(input_path)
            map(vis.remove_actor, vis.actors.keys())
            vis.setup()

        n = count(0)

        # peform camera actions and save outputs as required
        for action in args.camera_actions:
            # parse the action
            method, tup, save_after = parse_camera_action(action)
            print '%s(*%s), save_after=%s' % (method, tup, save_after)

            if ':' in method:
                name, method = method.split(':')
                vis.actor_properties(name, (method, tup))
            else:
                # execute the camera action
                vis.camera_actions((method, tup))

            # save if required
            if not interactive_session and save_after:
                full_path = os.path.join(output_paths[i], '%d.png' % next(n))
                print 'Output: ', full_path
                vis.write(full_path, magnification=args.magnification)

        # show if interactive
        if interactive_session:
            print 'Interactive'

            vis.ren_win.Render()
            vis.execute(magnification=args.magnification)

# test_tuple_from_string
def test_tuple_from_string():
    print tuple_from_string('1')

if __name__ == '__main__':
    main()
    # test_tuple_from_string()
    
