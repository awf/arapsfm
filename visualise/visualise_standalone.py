# visualise_standalone.py

# Imports
import os
import inspect
import argparse
from visualise import *
from itertools import ifilter, count, izip
from functools import wraps

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
        restrict_setup = kwargs.pop('restrict_setup', None)
        self.subdivide = kwargs.pop('subdivide', 0)
        self.compute_normals = kwargs.pop('compute_normals', 0)

        with_core = kwargs.pop('with_core', None)

        self.z = np.load(filename)

        if with_core is not None:
            self.core_V = np.load(with_core)['V']
        else:
            self.core_V = None

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
        if self.subdivide == 0:
            self.vis.add_mesh(self[self.vertices_key], 
                              self['T'],
                              self['L'],
                              compute_normals=self.compute_normals)
        else:
            T, V = self['T'], self[self.vertices_key]
            for i in xrange(self.subdivide):
                T, V = loop.subdivide(T, V)
                              
            self.vis.add_mesh(V, T, compute_normals=self.compute_normals)

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
        if len(X) != y.shape[0]:
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
    def _add_sectioned_arap(self):
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
    parser.add_argument('-a', dest='actor_properties', type=str, default=[],
                        action='append', help='actor properties')
    parser.add_argument('--magnification', type=int, default=1,
                        help='magnification')
    parser.add_argument('--vertices-key', type=str, default='V',
                        help='key to retrieve vertices')
    parser.add_argument('--output_image_first', 
                        action='store_true',
                        default=False)
    parser.add_argument('--with_core', type=str, default=None)
    parser.add_argument('--subdivide', type=int, default=0)
    parser.add_argument('--compute_normals', action='store_true', default=False)

    args = parser.parse_args()

    # echo arguments
    print 'Arguments:', args

    # setup visualisation
    print 'Source file: %s' % args.input
    print 'Available keys:', np.load(args.input).keys()

    restrict_setup = ('_add_image,') if args.output_image_first else None

    vis = StandaloneVisualisation(args.input,
                                  vertices_key=args.vertices_key,
                                  restrict_setup=restrict_setup,
                                  with_core=args.with_core,
                                  subdivide=args.subdivide,
                                  compute_normals=args.compute_normals)

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
    
