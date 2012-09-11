# run_test_mesh.py

# Imports
import numpy as np
from vtk_ import *
from test_mesh import *

# test_geometry_1
def test_geometry_1():
    T = np.array([[5, 1, 0],
                  [0, 1, 2],
                  [0, 2, 3],
                  [0, 3, 4],
                  [0, 4, 5]], dtype=np.int32)

    V = np.empty((6, 3), dtype=np.float64)
    V[0] = (0., 0., 0.)

    t = np.linspace(0, 2*np.pi, 5, endpoint=False)
    V[1:,0] = np.cos(t)
    V[1:,1] = np.sin(t)
    V[1:,2] = 0.

    return V, T

# test_geometry_2
def test_geometry_2():
    T = np.array([[0, 1, 2]], dtype=np.int32)

    V = np.empty((3, 3), dtype=np.float64)
    V[0] = (0., 0., 0.)
    V[1] = (0., 1., 0.)
    V[2] = (1., 0., 0.)

    return V, T

# test_geometry_3
def test_geometry_3():
    T = np.array([[0, 1, 3],
                  [1, 4, 3],
                  [1, 2, 4],
                  [2, 5, 4],
                  [3, 4, 6],
                  [6, 4, 7],
                  [4, 5, 7],
                  [5, 8, 7]], dtype=np.int32)

    V = np.empty((9, 3), dtype=np.float64)
    i = np.arange(3)
    V[:, 0] = np.tile(i, 3)
    V[:, 1] = 2. - np.repeat(i, 3)
    V[:, 2] = 0.

    return V, T

# load_model_2
def load_model_2():
    z = np.load('Models/CHIHUAHUA.npz')
    return z['V'], z['T']

# main_test_mesh
def main_test_mesh():
    V, T = test_geometry_3()
    test_mesh(V, T)

# nring_interface
def nring_interface(V, T, get_ring):
    T_ = faces_to_vtkCellArray(T)
    model_pd = numpy_to_vtkPolyData(V, T_)

    # model
    model_mapper = vtk.vtkPolyDataMapper()
    model_mapper.SetInput(model_pd)

    model_actor = vtk.vtkActor()
    model_actor.SetMapper(model_mapper)
    model_actor.GetProperty().SetColor(0.26, 0.58, 0.76)

    # ring (ring_pd)
    ring_points = vtk.vtkPoints()
    ring_pd = vtk.vtkPolyData()
    ring_pd.SetPoints(ring_points)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.05)

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInput(ring_pd)

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
    picker = vtk.vtkPointPicker()

    def annotate_pick(obj, event):
        i = picker.GetPointId()
        if i == -1: return

        ring = get_ring(i)
        print 'ring:', ring
        print 'unique(ring):', np.unique(ring)
        print 'len(ring):', len(ring)
        print 'len(unique(ring)):', len(np.unique(ring))

        ring_points.SetNumberOfPoints(len(ring))
        for j, i in enumerate(ring):
            ring_points.SetPoint(j, V[i])

        ring_pd.Modified()
        glyph_actor.VisibilityOn()

        ren_win.Render()

    picker.AddObserver('EndPickEvent', annotate_pick)
    iren.SetPicker(picker)

    iren.Initialize()
    iren.Start()

# main_test_nring
def main_test_nring():
    V, T = load_model_2()

    N = 2
    def get_ring(vertex):
        return get_nring(V, T, vertex, 2, True)

    nring_interface(V, T, get_ring)

if __name__ == '__main__':
    # main_test_mesh()
    main_test_nring()
