# check_rotation.py

# Imports
import argparse
import numpy as np
from visualise.visualise import VisualiseMesh
from util.cmdline import load_input_geometry, load_input_mesh
from geometry import quaternion, axis_angle
from misc.numpy_ import normalise
from matplotlib import cm

# main_linear_basis
def main_linear_basis():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('initial_geometry', type=str)
    parser.add_argument('--use_linear_transform', 
                        action='store_true',
                        default=False)
    parser.add_argument('axis', type=str)
    parser.add_argument('rotations', type=str)
    parser.add_argument('--in_radians', 
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    args.axis = eval(args.axis)
    args.rotations = eval(args.rotations)

    vis = VisualiseMesh()

    T = load_input_mesh(args.model)
    V = load_input_geometry(args.initial_geometry, args.use_linear_transform)
    V -= np.mean(V, axis=0)
    vis.add_mesh(V, T, actor_name='base', color=(31, 120, 180),
                 compute_normals=True)

    axis = np.asarray(args.axis)
    assert axis.ndim == 1 and axis.shape[0] == 3
    axis /= norm(axis)

    rotations = (args.rotations if args.in_radians
                                else map(np.deg2rad, args.rotations))

    q = map(lambda r: quaternion.quat(axis * r), rotations)
    R = map(quaternion.rotationMatrix, q)
    V1 = map(lambda r: np.dot(V, np.transpose(r)), R)

    n = len(V1)
    colors = cm.autumn(np.linspace(0., 1., n, endpoint=True),
                       bytes=True)

    for i in xrange(n):
        vis.add_mesh(V1[i], T, 
                     actor_name='rotated_%d' % i,
                     color=colors[i],
                     compute_normals=True)

    vis.camera_actions(('SetParallelProjection', (True,)))

    for actor in vis.actors.keys():
        vis.actor_properties(actor, ('SetRepresentation', (3,)))

    vis.execute()

# main_compositional_basis
def main_compositional_basis():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('initial_geometry', type=str)
    parser.add_argument('--use_linear_transform', 
                        action='store_true',
                        default=False)

    parser.add_argument('axes', type=str)
    parser.add_argument('rotations', type=str)
    parser.add_argument('--in_radians', 
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    for key in ['axes', 'rotations']:
        setattr(args, key, eval(getattr(args, key)))

    vis = VisualiseMesh()

    T = load_input_mesh(args.model)
    V = load_input_geometry(args.initial_geometry, args.use_linear_transform)
    V -= np.mean(V, axis=0)
    vis.add_mesh(V, T, actor_name='base', color=(31, 120, 180),
                 compute_normals=True)

    axes = np.asarray(args.axes)
    assert axes.shape == (2, 3)
    axes = normalise(axes)

    rotations = np.atleast_2d(args.rotations)
    assert rotations.shape[1] == 2

    if args.in_radians:
        rotations = np.rad2deg(rotations)

    n = len(rotations)
    colors = cm.autumn(np.linspace(0., 1., n, endpoint=True),
                       bytes=True)

    for i in xrange(n):
        x = reduce(axis_angle.axAdd, axes * rotations[i][:, np.newaxis])
        q = quaternion.quat(x)
        R = quaternion.rotationMatrix(q)
        V1 = np.dot(V, np.transpose(R))
        vis.add_mesh(V1, T, actor_name='rotation_%d' % i, 
                     color=colors[i],
                     compute_normals=True)

    vis.execute()

if __name__ == '__main__':
    main_compositional_basis()

