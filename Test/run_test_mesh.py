# run_test_mesh.py

# Imports
import numpy as np
from visualise import visualise
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

# triangles_at_vertex_interface
def triangles_at_vertex_interface(V, T, get_faces):
    T_ = faces_to_vtkCellArray(T)
    model_pd = numpy_to_vtkPolyData(V, T_)

    L = []

    # setup the programmable filter to color specific faces
    color_face = vtk.vtkProgrammableFilter()
    color_face.SetInput(model_pd)

    def color_face_callback():
        input_ = color_face.GetInput()
        output = color_face.GetOutput()
        output.ShallowCopy(input_)

        color_lookup = vtk.vtkIntArray()
        color_lookup.SetNumberOfValues(model_pd.GetNumberOfPolys())

        npy_cl = vtk_to_numpy(color_lookup)
        npy_cl.fill(0)

        if L is not None:
            npy_cl[L] = 1

        output.GetCellData().SetScalars(color_lookup)

    color_face.SetExecuteMethod(color_face_callback)

    # setup the lookup table for coloring the different cells
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.Build()

    lut.SetTableValue(0, *visualise.int2dbl(31, 120, 180))
    lut.SetTableValue(1, *visualise.int2dbl(178, 223, 138))

    # model
    model_mapper = vtk.vtkPolyDataMapper()
    model_mapper.SetInputConnection(color_face.GetOutputPort())
    model_mapper.SetScalarRange(0, 1)
    model_mapper.SetLookupTable(lut)

    model_actor = vtk.vtkActor()
    model_actor.SetMapper(model_mapper)
    model_actor.GetProperty().SetColor(0.26, 0.58, 0.76)

    # renderer
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.AddActor(model_actor)

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

        faces = get_faces(i)
        print 'faces:', faces

        del L[:]
        L.extend(faces)
        print 'L:', L

        # force update
        model_pd.Modified()
        ren_win.Render()

    picker.AddObserver('EndPickEvent', annotate_pick)
    iren.SetPicker(picker)

    iren.Initialize()
    iren.Start()

# main_test_triangles_at_vertex
def main_test_triangles_at_vertex():
    V, T = load_model_2()
    
    def get_faces(vertex):
        return get_triangles_at_vertex(V, T, vertex)

    triangles_at_vertex_interface(V, T, get_faces)

if __name__ == '__main__':
    # main_test_mesh()
    # main_test_nring()
    main_test_triangles_at_vertex()
