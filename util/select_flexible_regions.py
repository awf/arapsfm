# select_flexible_regions.py

# Imports
import vtk
from PyQt4 import QtCore, QtGui
from PyQt4_.vtk.QVTKWidget import QVTKWidget
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtk_ import numpy_to_vtkPolyData, iter_vtkCollection

import os
import numpy as np

from mesh.box_model import box_model
from mesh import faces

from pprint import pprint
from matplotlib import cm

# test_vtkInteractorStyleRubberBandPick
def test_vtkInteractorStyleRubberBandPick():
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)

    V, T = box_model(5, 10, 1.0, 1.0) 
    poly_data = numpy_to_vtkPolyData(V, faces.faces_to_cell_array(T))

    idFilter = vtk.vtkIdFilter()
    idFilter.SetInput(poly_data)
    idFilter.SetIdsArrayName('i')
    idFilter.Update()

    poly_data = idFilter.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(poly_data)
    mapper.SetScalarVisibility(False)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.0, 0.6, 0.3)

    ren.AddActor(actor)

    visible = vtk.vtkSelectVisiblePoints()
    visible.SetInput(poly_data)
    visible.SetRenderer(ren)

    # highlight
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.2)

    highlight_poly_data = vtk.vtkPolyData()

    highlight_glyph = vtk.vtkGlyph3D()
    highlight_glyph.SetInput(highlight_poly_data)
    highlight_glyph.SetSourceConnection(sphere.GetOutputPort())

    highlight_mapper = vtk.vtkPolyDataMapper()
    highlight_mapper.SetInputConnection(highlight_glyph.GetOutputPort())

    highlight_actor = vtk.vtkActor()
    highlight_actor.SetMapper(highlight_mapper)
    highlight_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    highlight_actor.VisibilityOff()
    highlight_actor.PickableOff()
    ren.AddActor(highlight_actor)

    # render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(ren)
    render_window.SetSize(400, 400)

    picker = vtk.vtkAreaPicker()

    def pickCallback(obj, event):
        props = obj.GetProp3Ds()
        if props.GetNumberOfItems() <= 0:
            return

        extract_geometry = vtk.vtkExtractGeometry()
        extract_geometry.SetImplicitFunction(picker.GetFrustum())
        extract_geometry.SetInput(props.GetLastProp3D().GetMapper().GetInput())
        extract_geometry.Update()

        unstructured_grid = extract_geometry.GetOutput()

        if unstructured_grid.GetPoints().GetNumberOfPoints() <= 0:
            return

        visible.Update()
        if visible.GetOutput().GetPoints().GetNumberOfPoints() <= 0:
            return

        i = np.intersect1d(
            vtk_to_numpy(unstructured_grid.GetPointData().GetArray('i')),
            vtk_to_numpy(visible.GetOutput().GetPointData().GetArray('i')))

        if i.shape[0] <= 0:
            return

        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(i.shape[0])
        vtk_points_data = vtk_to_numpy(vtk_points.GetData())
        vtk_points_data.flat = np.require(V[i], np.float32, 'C')

        highlight_poly_data.SetPoints(vtk_points)
        highlight_actor.VisibilityOn()

    picker.AddObserver('EndPickEvent', pickCallback)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)
    iren.SetPicker(picker)

    ren.ResetCamera()

    render_window.Render()

    style = vtk.vtkInteractorStyleRubberBandPick()
    style.SetCurrentRenderer(ren)
    iren.SetInteractorStyle(style)

    iren.Initialize()
    iren.Start()

# Constants
COLORMAP = cm.Set1_r(np.linspace(0., 1., 9))
LUT = vtk.vtkLookupTable()
LUT.SetNumberOfTableValues(COLORMAP.shape[0])
LUT.Build()
for i, c in enumerate(COLORMAP):
    LUT.SetTableValue(i, *c)

# MeshView
class MeshView(QVTKWidget):
    def __init__(self, parent=None):
        QVTKWidget.__init__(self, parent)
        self._setup_pipeline()

    def _setup_pipeline(self):
        ren = vtk.vtkRenderer()
        ren.SetBackground(1.0, 1.0, 1.0)

        color_points = vtk.vtkProgrammableFilter()
        def callback_color_points():
            input_ = color_points.GetInput()
            output = color_points.GetOutput()
            output.ShallowCopy(input_)

            lookup = vtk.vtkFloatArray()
            lookup.SetNumberOfValues(input_.GetNumberOfPoints())
            npy_lookup = vtk_to_numpy(lookup)
            npy_lookup.flat = self.k.astype(np.float32) / (COLORMAP.shape[0] - 1)

            output.GetPointData().SetScalars(lookup)
        color_points.SetExecuteMethod(callback_color_points)

        source = vtk.vtkSphereSource()

        vertices_glyph = vtk.vtkGlyph3D()
        vertices_glyph.SetInputConnection(color_points.GetOutputPort())
        vertices_glyph.SetSourceConnection(source.GetOutputPort())
        vertices_glyph.SetScaleModeToDataScalingOff()
        vertices_glyph.SetColorModeToColorByScalar()

        vertices_mapper = vtk.vtkPolyDataMapper()
        vertices_mapper.SetInputConnection(vertices_glyph.GetOutputPort())
        vertices_mapper.SetLookupTable(LUT)

        vertices_actor = vtk.vtkActor()
        vertices_actor.SetMapper(vertices_mapper)
        vertices_actor.GetProperty().SetColor(*COLORMAP[0, :3])
        vertices_actor.PickableOff()
        ren.AddActor(vertices_actor)

        color_faces = vtk.vtkProgrammableFilter()
        def callback_color_faces():
            input_ = color_faces.GetInput()
            output = color_faces.GetOutput()
            output.ShallowCopy(input_)

            lookup = vtk.vtkFloatArray()
            lookup.SetNumberOfValues(input_.GetNumberOfPolys())
            npy_lookup = vtk_to_numpy(lookup)

            labelled_T = self.k[self.T]
            for i in xrange(input_.GetNumberOfPolys()):
                l = np.argmax(np.bincount(labelled_T[i]))
                npy_lookup[i] = float(l) / (COLORMAP.shape[0] - 1)

            output.GetCellData().SetScalars(lookup)
        color_faces.SetExecuteMethod(callback_color_faces)

        model_mapper = vtk.vtkPolyDataMapper()
        model_mapper.SetInputConnection(color_faces.GetOutputPort())
        model_mapper.SetLookupTable(LUT)

        model_actor = vtk.vtkActor()
        model_actor.SetMapper(model_mapper)
        model_actor.GetProperty().SetColor(*COLORMAP[0, :3])
        ren.AddActor(model_actor)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(ren)
        render_window.SetSize(400, 400)

        iren = self.GetInteractor()
        iren.SetRenderWindow(render_window)

        style = vtk.vtkInteractorStyleRubberBandPick()
        style.SetCurrentRenderer(ren)
        iren.SetInteractorStyle(style)

        self.pipeline = locals()

        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

    def add_mesh(self, V, T, **kwargs):
        self.V = V
        self.T = T
        self.k = np.zeros(V.shape[0], dtype=np.int32)

        enorm = lambda V: np.sqrt(np.sum(V*V, axis=-1))
        edge_lengths = np.r_[enorm(V[T[:,0]] - V[T[:,1]]),
                             enorm(V[T[:,1]] - V[T[:,2]]),
                             enorm(V[T[:,2]] - V[T[:,0]])]

        self.pipeline['source'].SetRadius(0.3 * np.amin(edge_lengths))

        model_poly_data = numpy_to_vtkPolyData(V, faces.faces_to_cell_array(T))
        self.pipeline['color_points'].SetInput(model_poly_data)
        self.pipeline['color_faces'].SetInput(model_poly_data)

        self.pipeline['ren'].ResetCamera()
        self.GetRenderWindow().Render()

# main
def main():
    V, T = box_model(5, 15, 1.0, 1.0) 

    qapp = QtGui.QApplication([])

    view = MeshView()
    view.add_mesh(V, T)
    view.show()

    qapp.exec_()

if __name__ == '__main__':
    # test_vtkInteractorStyleRubberBandPick()
    main()

