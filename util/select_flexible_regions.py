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

if __name__ == '__main__':
    test_vtkInteractorStyleRubberBandPick()
