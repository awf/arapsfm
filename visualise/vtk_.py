# vtk_.py

# Imports
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk 

# Constants
ACTOR_OPT = dict(Color=(0.26, 0.58, 0.76),
                 Lighting=True,)

# view_vtkPolyData
def view_vtkPolyData(poly_data, highlight=None, actor_opt=None, 
    camera_opt={}, filename=None, sphere_radius=0.05):
    if actor_opt is None:
        actor_opt = ACTOR_OPT

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor_property = actor.GetProperty()

    for property_, args in actor_opt.iteritems():
        method_name = 'Set%s' % property_
        try:
            method = getattr(actor_property, method_name)
        except :
            raise ValueError('vtkActor has no method "%s"' % method_name)

        method(args)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(1.0, 1.0, 1.0)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(ren)
    render_window.SetSize(800, 800)

    vertex_sphere = vtk.vtkSphereSource()
    vertex_sphere.SetRadius(sphere_radius)

    vertex_mapper = vtk.vtkPolyDataMapper()

    vertex_actor = vtk.vtkActor()
    vertex_actor.SetMapper(vertex_mapper)
    vertex_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    vertex_actor.PickableOff()

    if highlight is not None:
        hl_points = vtk.vtkPoints()
        hl_points.SetNumberOfPoints(len(highlight))
        for i, p in enumerate(highlight):
            hl_point = poly_data.GetPoints().GetPoint(p)
            hl_points.SetPoint(i, *hl_point)

        hl_poly_data = vtk.vtkPolyData()
        hl_poly_data.SetPoints(hl_points)

        vertex_glyph = vtk.vtkGlyph3D()
        vertex_glyph.SetInput(hl_poly_data)
        vertex_glyph.SetSourceConnection(vertex_sphere.GetOutputPort())

        vertex_mapper.SetInputConnection(vertex_glyph.GetOutputPort())
    else:
        vertex_mapper.SetInputConnection(vertex_sphere.GetOutputPort())
        vertex_actor.VisibilityOff()

    ren.AddActor(vertex_actor)

    camera = ren.GetActiveCamera()

    _camera_opt = {'FocalPoint':(0,0,0), 'Position':(0, 0, 1),
                   'ViewUp':(0,1,0)}
    _camera_opt.update(camera_opt)

    for property_, args in _camera_opt.iteritems():
        method_name = 'Set%s' % property_
        try:
            method = getattr(camera, method_name)
        except :
            raise ValueError('vtkCamera has no method "%s"' % method_name)

        method(args)

    ren.ResetCamera()
        
    if filename is not None:
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(render_window)
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.SetFileName(filename)

        render_window.Render()
        writer.Write()
        return

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    if highlight is None:
        picker = vtk.vtkPointPicker()

        def annotatePick(obj, event):
            point_id = picker.GetPointId()
            position = poly_data.GetPoints().GetPoint(point_id)
            vertex_sphere.SetCenter(position)
            vertex_actor.VisibilityOn()
            print '%d:' % point_id, position

        picker.AddObserver('EndPickEvent', annotatePick)

        iren.SetPicker(picker)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetCurrentRenderer(ren)
    iren.SetInteractorStyle(style)

    iren.Initialize()
    iren.Start()

# vtkPolyData_to_numpy
def vtkPolyData_to_numpy(poly_data):
    points = vtk_to_numpy(poly_data.GetPoints().GetData())
    cells = vtk_to_numpy(poly_data.GetPolys().GetData())
    
    return points, cells
    
# numpy_to_vtkPolyData
def numpy_to_vtkPolyData(points, cells):
    # construct cell array
    vtk_cells = vtk.vtkCellArray()

    # if cells is list of faces
    if isinstance(cells, list):
        for cell in cells:
            n = len(cell)
            vtk_cells.InsertNextCell(n)

            for id_ in cell:
                vtk_cells.InsertCellPoint(id_)

    # otherwise treat as single dimension array
    # this is the output of `vtkPolyData_to_numpy`
    elif isinstance(cells, np.ndarray):
        i = 0
        while i < cells.shape[0]:
            n = cells[i]
            vtk_cells.InsertNextCell(n)

            for j in range(n):
                vtk_cells.InsertCellPoint(cells[(i+1) + j])

            i += n + 1

    else:
        raise ValueError('unable to handle cells of type: %s' %
                         type(cells))

    # copy points directly using a numpy iterator over the memory block
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(points.shape[0])
    npy_points = vtk_to_numpy(vtk_points.GetData())
    npy_points.flat = np.asarray(points, dtype=np.float32)

    poly_data = vtk.vtkPolyData()

    poly_data.SetPoints(vtk_points)
    poly_data.SetPolys(vtk_cells)
            
    return poly_data

# setup_vtkPolyData
def setup_vtkPolyData(p, faces):
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(p.shape[0])
    for i, q in enumerate(p):
        points.SetPoint(i, *q)

    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for f in face:
            cells.InsertCellPoint(f)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(cells)

    return poly_data

# reduce_vtkPolyData
def reduce_vtkPolyData(poly_data, reduction=None):
    decimate = vtk.vtkQuadricDecimation() 
    if reduction is not None:
        decimate.SetTargetReduction(reduction)
    decimate.SetInput(poly_data)
    decimate.Update()
    
    return decimate.GetOutput()

# clean_vtkPolyData
def clean_vtkPolyData(poly_data):
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInput(poly_data)

    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(triangle_filter.GetOutputPort())
    clean_filter.SetTolerance(0.0)
    clean_filter.Update()

    return clean_filter.GetOutput()

# faces_to_vtkCellArray
def faces_to_vtkCellArray(faces):
    N = len(faces) + np.sum(map(len, faces))
    cells = np.empty(N, dtype=np.int32)

    i = 0
    for face in faces:
        n = len(face)
        cells[i] = n
        cells[i+1:i+n+1] = face

        i += n + 1

    return cells

# iter_vtkCellArray
def iter_vtkCellArray(cells):
    j = 0
    while j < cells.shape[0]:
        n = cells[j]
        yield cells[j+1:n+j+1]
        j += n + 1

# vtkMatrix4x4_to_numpy
def vtkMatrix4x4_to_numpy(mat):
    T = np.empty((4,4), dtype=float)
    for i in xrange(4):
        for j in xrange(4):
            T[i,j] = mat.GetElement(i,j)

    return T

# numpy_to_vtkMatrix4x4
def numpy_to_vtkMatrix4x4(T):
    mat = vtk.vtkMatrix4x4()
    for i in xrange(4):
        for j in xrange(4):
            mat.SetElement(i, j, T[i,j])
   
    return mat


