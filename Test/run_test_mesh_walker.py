# run_test_mesh_walker.py

# Imports
import numpy as np
from vtk_ import *
from test_mesh_walker import *

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import norm

# Barycentric conversion

# make_bary
def make_bary(u):
    u = u[:2]
    return np.r_[u, 1.0 - np.sum(u)]

# bary2pos
def bary2pos(V, u):
    return np.dot(u, V)

# pos2bary
def pos2bary(V, p):
    n = np.cross(V[0] - V[2], V[1] - V[2])
    na = np.cross(V[1] - V[0], p - V[0])
    nb = np.cross(V[2] - V[1], p - V[1])
    sqrl_n = norm(n)**2

    u = np.empty(3, dtype=np.float64)
    u[0] = np.dot(n, nb) / sqrl_n
    u[2] = np.dot(n, na) / sqrl_n
    u[1] = 1.0 - u[0] - u[2]

    return u

# test_bary_conversion
def test_bary_conversion():
    # trial basis
    # V = np.array([[-0.5, -np.sqrt(3)/4],
    #               [0.5,  -np.sqrt(3)/4],
    #               [0, np.sqrt(3)/4]], dtype=np.float64)
    V = np.random.randn(9).reshape(3,3)

    u = make_bary([0.7, 0.3])
    p = bary2pos(V, u)
    u1 = pos2bary(V, p)
    print u
    print p
    print u1

    f, ax = plt.subplots()
    x, y, z = np.transpose(np.r_['0,2', V, V[0]])
    ax.plot(x, y, 'k-')
    ax.plot(p[0], p[1], 'ro')
    cols = 'rgb'
    for i, v in enumerate(V):
        ax.plot(v[0], v[1], '%so' % cols[i])
    plt.show()

# walk_interface
def walk_interface(V, T, walker):
    T_ = faces_to_vtkCellArray(T)
    model_pd = numpy_to_vtkPolyData(V, T_)

    # model
    model_mapper = vtk.vtkPolyDataMapper()
    model_mapper.SetInput(model_pd)

    model_actor = vtk.vtkActor()
    model_actor.SetMapper(model_mapper)
    model_actor.GetProperty().SetColor(0.26, 0.58, 0.76)

    # walk direction (direction_pd)
    direction_points = vtk.vtkPoints()
    direction_points.SetNumberOfPoints(2)

    direction_lines = vtk.vtkCellArray()
    direction_lines.InsertNextCell(2)
    direction_lines.InsertCellPoint(0)
    direction_lines.InsertCellPoint(1)

    direction_pd = vtk.vtkPolyData()
    direction_pd.SetPoints(direction_points)
    direction_pd.SetLines(direction_lines)

    direction_tube = vtk.vtkTubeFilter()
    direction_tube.SetInput(direction_pd)
    direction_tube.SetRadius(0.02)

    direction_mapper = vtk.vtkPolyDataMapper()
    direction_mapper.SetInputConnection(direction_tube.GetOutputPort())

    direction_actor = vtk.vtkActor()
    direction_actor.SetMapper(direction_mapper)
    direction_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
    direction_actor.VisibilityOff()
    direction_actor.PickableOff()

    # walk path (path_pd)
    path_points = vtk.vtkPoints()
    path_pd = vtk.vtkPolyData()
    path_pd.SetPoints(path_points)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.05)

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInput(path_pd)

    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyph.GetOutputPort())

    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    glyph_actor.VisibilityOff()
    glyph_actor.PickableOff()

    # renderer
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.AddActor(model_actor)
    ren.AddActor(glyph_actor)
    ren.AddActor(direction_actor)

    # render window
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(600, 600)

    # render window interaction
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetCurrentRenderer(ren)
    iren.SetInteractorStyle(style)

    # interface
    picker = vtk.vtkCellPicker()

    def annotate_pick(obj, event):
        i = picker.GetCellId()
        if i == -1: return

        # resolve parametric coordinates (relative to Vi[0]) to world
        # coordinates
        Ti = T[i]
        Vi = V[Ti]
        B = Vi[1:] - Vi[0]
        p = picker.GetPCoords()
        v0 = np.dot(p[:2], B) + Vi[0]
        print 'Cell Id:', i
        print 'Vi:', np.around(Vi, decimals=0)
        print 'v0:', np.around(v0, decimals=1)

        # resolve barycentric coordinates
        u = pos2bary(Vi, v0)
        print 'u:', np.around(u, decimals=2)

        # set delta which is applied to u (must sum to 0)
        delta = np.random.randn(2).astype(np.float64)

        world_delta, path = walker(u[:2], i, delta)

        v1 = v0 + world_delta
        direction_points.SetPoint(0, *v0)
        direction_points.SetPoint(1, *v1)

        direction_pd.Modified()
        direction_actor.VisibilityOn()

        # set the path
        path_points.SetNumberOfPoints(len(path))
        for j, (i, u) in enumerate(path):
            v = bary2pos(V[T[i]], make_bary(u))
            path_points.SetPoint(j, *v)

        path_pd.Modified()
        glyph_actor.VisibilityOn()

        ren_win.Render()

        """
        # execute the walker 
        path, first_world_delta = walker(u[:2], i, delta)
        pprint(path)

        if test_walker is not None:
            U = np.empty((1, 2), dtype=np.float64)
            U[0] = u[:2]
            face_lookup = np.array([i], dtype=np.int32)
            delta_ = np.empty((1, 2), dtype=np.float64)
            delta_[0] = delta
            test_walker(U, face_lookup, delta_)
            print 'test_walker: face_lookup:', face_lookup
            print 'test_walker: U:', U
            print 'path[-1]:', path[-1]

        # set the path
        path_points.SetNumberOfPoints(len(path))
        for j, (i, u) in enumerate(path):
            v = bary2pos(V[T[i]], u)
            path_points.SetPoint(j, *v)

        # set the implied world delta if available
        if first_world_delta is not None:
            v1 = v0 + first_world_delta
            direction_points.SetPoint(0, *v0)
            direction_points.SetPoint(1, *v1)

            direction_pd.Modified()
            direction_actor.VisibilityOn()
        else:
            direction_actor.VisibilityOff()
            
        path_pd.Modified()
        glyph_actor.VisibilityOn()
        ren_win.Render()
        """

    picker.AddObserver('EndPickEvent', annotate_pick)
    iren.SetPicker(picker)

    iren.Initialize()
    iren.Start()

# load_model_2
def load_model_2():
    z = np.load('Models/CHIHUAHUA.npz')
    return z['V'], z['T']

# wrapped_apply_displacement
def wrapped_apply_displacement(V, T):
    V = np.require(V, dtype=np.float64, requirements='C')
    T = np.require(T, dtype=np.int32, requirements='C')

    def walker(u, l, delta):
        # resolve world delta
        delta1 = np.r_[delta, - np.sum(delta)]
        world_delta = bary2pos(V[T[l]], delta1)

        delta = np.require([delta], dtype=np.float64, requirements='C')
        U = np.require([u], dtype=np.float64, requirements=['C', 'O', 'W'])
        L = np.require([l], dtype=np.int32, requirements=['C', 'O', 'W'])

        # apply displacement
        apply_displacement(V, T, U, L, delta)

        # return start and end
        path = [(l, u), (L[0], U[0])]

        return world_delta, path

    return walker

# main
def main():
    # raise on all NumPy errors
    np.seterr(all='raise')

    # load model
    V, T = load_model_2()

    walker = wrapped_apply_displacement(V, T)

    walk_interface(V, T, walker)

if __name__ == '__main__':
    main()
