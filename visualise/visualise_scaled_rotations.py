# visualise_scaled_rotations.py

# Imports
import numpy as np
import argparse
from visualise import *
from mesh import weights
from mesh.faces import faces_to_cell_array
from itertools import groupby, count
from operator import itemgetter
from geometry import quaternion as quat
from geometry.axis_angle import *
from solvers.arap import ARAPVertexSolver
from matplotlib import cm

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('aux_scales', type=str)
    parser.add_argument('-c', dest='camera_actions', type=str, default=[],
                        action='append', help='camera actions')
    parser.add_argument('-o', dest='output_directory', type=str, default=None,
                        help='output directory')
    parser.add_argument('--magnification', type=int, default=1,
                        help='magnification')

    args = parser.parse_args()
    z = np.load(args.input)

    vis = VisualiseMesh()

    # V0 (purple)
    V0 = z['V0']
    vis.add_mesh(V0, z['T'], actor_name='V0')
    lut = vis.actors['V0'].GetMapper().GetLookupTable()
    lut.SetTableValue(0, *int2dbl(255, 0, 255))

    # setup `solveV0_X`
    T_ = faces_to_cell_array(z['T'])
    adj, W = weights.weights(V0, T_, weights_type='cotan')
    solveV0_X = ARAPVertexSolver(adj, W, V0)

    # show additional scales
    aux_scales = eval(args.aux_scales)
    print 'aux_scales: ', aux_scales
    
    rotM = lambda x: quat.rotationMatrix(quat.quat(x))

    N = len(aux_scales)
    cmap = cm.jet(np.linspace(0., 1., N, endpoint=True))

    for i, scale in enumerate(aux_scales):
        print 'scale:', scale
        X = map(lambda x: axScale(scale, x), z['X'])
        V0_X = solveV0_X(map(rotM, X))

        actor_name = 'V0_X_%d' % i
        print 'actor_name:', actor_name
        vis.add_mesh(V0_X, z['T'], actor_name=actor_name)
        lut = vis.actors[actor_name].GetMapper().GetLookupTable()
        lut.SetTableValue(0, *cmap[i,:3])
        vis.actor_properties(actor_name, ('SetRepresentation', (3,)))
        
    # input frame
    vis.add_image(z['input_frame'])

    # is visualisation interface or to file?
    interactive_session = args.output_directory is None

    # setup output directory
    if not interactive_session and not os.path.exists(args.output_directory):
        print 'Creating directory: ', args.output_directory
        os.makedirs(args.output_directory)

    # n is the index for output files
    n = count(0)

    # apply camera actions sequentially
    for action in args.camera_actions:
        method, tup, save_after = parse_camera_action(action)
        print '%s(*%s), save_after=%s' % (method, tup, save_after)

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
    
if __name__ == '__main__':
    main()
