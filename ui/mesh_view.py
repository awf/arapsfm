# mesh_view.py

# Imports
from vtk_ import *

from PyQt4 import QtGui, QtCore
from PyQt4_.imageview import ImageView
from PyQt4_.vtk.QVTKWidget import QVTKWidget

# InteractiveMeshView
class InteractiveMeshView(QVTKWidget):
    pickChanged = QtCore.pyqtSignal(name='pickChanged')

    def __init__(self, parent=None):
        QVTKWidget.__init__(self, parent)

        self.poly_data = None
        self.has_image = False
        self.C = []

        self.last_point = [None, None]

        self.last_transform = None

        self._setup_pipeline()

        self.setSizePolicy(QtGui.QSizePolicy.Expanding, 
                           QtGui.QSizePolicy.Expanding)
        
    # Initialisation
    def _setup_pipeline(self):
        # mesh actor
        mapper = vtk.vtkPolyDataMapper()
        self.mapper = mapper

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.26, 0.58, 0.76)
        actor.GetProperty().SetLighting(True)

        self.actor = actor
        self.actor.SetVisibility(False)

        # auxiliary mesh actor
        aux_mapper = vtk.vtkPolyDataMapper()
        self.aux_mapper = aux_mapper

        aux_actor = vtk.vtkActor()
        aux_actor.SetMapper(aux_mapper)
        aux_actor.GetProperty().SetColor(0.76, 0.26, 0.58)
        aux_actor.GetProperty().SetLighting(True)
        self.aux_actor = aux_actor

        # selection actors
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(5.0)

        # mesh vertex selection
        sel_poly_data = vtk.vtkPolyData()
        sel_points = vtk.vtkPoints()
        sel_points.SetNumberOfPoints(1)
        sel_poly_data.SetPoints(sel_points)

        # image selection
        im_sel_poly_data = vtk.vtkPolyData()
        im_sel_points = vtk.vtkPoints()
        im_sel_points.SetNumberOfPoints(1)
        im_sel_poly_data.SetPoints(im_sel_points)

        sel_glyph = vtk.vtkGlyph3D()
        sel_glyph.SetInput(sel_poly_data)
        sel_glyph.SetSourceConnection(sphere.GetOutputPort())
        sel_glyph.SetScaleFactor(1.0)

        im_sel_glyph = vtk.vtkGlyph3D()
        im_sel_glyph.SetInput(im_sel_poly_data)
        im_sel_glyph.SetSourceConnection(sphere.GetOutputPort())
        im_sel_glyph.SetScaleFactor(1.0)

        sel_mapper = vtk.vtkPolyDataMapper()
        sel_mapper.SetInputConnection(sel_glyph.GetOutputPort())

        im_sel_mapper = vtk.vtkPolyDataMapper()
        im_sel_mapper.SetInputConnection(im_sel_glyph.GetOutputPort())

        sel_actor = vtk.vtkActor()
        sel_actor.SetMapper(sel_mapper)
        sel_actor.GetProperty().SetColor(1., 0., 0.)
        sel_actor.SetVisibility(False)
        sel_actor.PickableOff()

        im_sel_actor = vtk.vtkActor()
        im_sel_actor.SetMapper(im_sel_mapper)
        im_sel_actor.GetProperty().SetColor(1., 0., 0.)
        im_sel_actor.SetVisibility(False)
        im_sel_actor.PickableOff()

        self.sel_poly_data = sel_poly_data
        self.sel_actor = sel_actor

        self.im_sel_poly_data = im_sel_poly_data
        self.im_sel_actor = im_sel_actor

        # projection actor
        proj_points = vtk.vtkPoints()
        proj_points.SetNumberOfPoints(0)
        proj_lines = vtk.vtkCellArray()

        proj_poly_data = vtk.vtkPolyData()
        proj_poly_data.SetPoints(proj_points)
        proj_poly_data.SetLines(proj_lines)

        proj_tube = vtk.vtkTubeFilter()
        proj_tube.SetInput(proj_poly_data)
        proj_tube.SetRadius(3.0)
        proj_tube.SetNumberOfSides(8)

        proj_mapper = vtk.vtkPolyDataMapper()
        proj_mapper.SetInputConnection(proj_tube.GetOutputPort())

        proj_actor = vtk.vtkActor()
        proj_actor.SetMapper(proj_mapper)
        proj_actor.GetProperty().SetColor(1.0, 1.0, 0)
        proj_actor.GetProperty().SetLighting(True)
        proj_actor.PickableOff()
        proj_actor.SetVisibility(False)

        self.proj_poly_data = proj_poly_data
        self.proj_actor = proj_actor

        # image slice viewer
        reader = vtk.vtkPNGReader()
        self.reader = reader

        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputConnection(reader.GetOutputPort())
        image_actor.SetVisibility(False)
        self.image_actor = image_actor

        # renderer
        ren = vtk.vtkRenderer()
        ren.AddActor(actor)
        ren.AddActor(sel_actor)
        ren.AddActor(im_sel_actor)
        ren.AddActor(proj_actor)
        ren.AddActor(image_actor)
        ren.AddActor(aux_actor)
        ren.SetBackground(1., 1., 1.)
        self.ren = ren

        # render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(ren)
        self.SetRenderWindow(render_window)

        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.01)
        picker.AddObserver('EndPickEvent', self._observe_pick)

        iren = self.GetInteractor()
        iren.SetPicker(picker)

        # box widget
        transform = vtk.vtkTransform()
        def TransformActor(obj, event):
            obj.GetTransform(transform)
            self._transform_update()

        box_widget = vtk.vtkBoxWidget()
        box_widget.SetPlaceFactor(1.00)
        box_widget.SetInteractor(iren)
        box_widget.SetProp3D(actor)

        box_widget.AddObserver("InteractionEvent", TransformActor)

        self.box_widget = box_widget
        self.box_transform = transform

        style = vtk.vtkInteractorStyleImage()
        style.SetInteractionModeToImage3D()
        style.SetCurrentRenderer(ren)
        iren.SetInteractorStyle(style)

    # Callbacks
    def _observe_pick(self, picker, event):
        if self.poly_data is None or not self.has_image:
            return

        i = picker.GetPointId()
        if i < 0:
            return 

        if picker.GetProp3D() is self.actor:
            self._set_point(i)
        elif picker.GetProp3D() is self.image_actor:
            y, x = divmod(i, self.image_actor.GetMaxXBound() + 1)
            self._set_image_point(x, y)

        self.pickChanged.emit()

    def _set_image_point(self, x , y):
        sel_points = self.im_sel_poly_data.GetPoints()
        sel_points.SetPoint(0, x, y, 0)
        self.im_sel_poly_data.Modified()

        self.last_point[1] = (x, y)
        self.im_sel_actor.SetVisibility(True)
        self.GetRenderWindow().Render()

    def _set_point(self, i=None):
        if i is None:
            if self.last_point[0] is None:
                return

            i = self.last_point[0]

        self.last_point[0] = i

        sel_points = self.sel_poly_data.GetPoints()
        sel_points.SetPoint(0, *self.V[i])
        self.sel_poly_data.Modified()

        self.sel_actor.SetVisibility(True)
        self.GetRenderWindow().Render()

    def _transform_update(self):
        T = self.transform()

        # update main vertices
        self.V[:] = np.dot(self.V0_prime, np.transpose(T[:3,:3])) + T[:3,-1]
        self.poly_data.Modified()

        # update selection
        self._set_point()

        # update projection
        self._projection_update()

        # reset deformation
        self.last_transform = None

    def _projection_update(self, project_to_aux=False):
        if project_to_aux:
            V, _ = vtkPolyData_to_numpy(self.aux_poly_data)
        else:
            V = self.V

        proj_points = self.proj_poly_data.GetPoints()

        for j in xrange(len(self.C)):
            (x, y), i = self.C[j]
            proj_points.SetPoint(2*j + 1, V[i])

        self.proj_actor.SetVisibility(True)
        self.proj_poly_data.Modified()
        self.GetRenderWindow().Render()

    # Setters / Getters
    def point(self):
        return self.last_point

    def transform(self):
        mat = self.box_transform.GetMatrix()
        if mat is None:
            T = np.eye(4)
        else:
            T = vtkMatrix4x4_to_numpy(mat)
        return T

    def triangles(self):
        if self.poly_data is None:
            return None

        V, T_ = vtkPolyData_to_numpy(self.poly_data)
        return np.asarray(list(iter_vtkCellArray(T_)), dtype=np.int32)
        
    def reset(self):
        # reset poly data to original points
        if self.poly_data is not None:
            self.V[:] = self.V0[:]
            self.V0_prime[:] = self.V0[:]
            self.poly_data.Modified()

        # reset the transformation
        self.set_transform(np.eye(4))

    def set_image(self, filename):
        self.reader.SetFileName(filename)
        self.image_actor.SetVisibility(True)

        camera = self.ren.GetActiveCamera()
        camera.SetFocalPoint(0,0,0)
        camera.SetPosition(0, 0, 1)
        camera.SetViewUp(0, 1, 0)
        camera.SetParallelProjection(True)

        self.ren.ResetCamera()
        self.GetRenderWindow().Render()

        self.has_image = True
        
    def set_polydata(self, poly_data):
        if self.has_image:
            V, cells = vtkPolyData_to_numpy(poly_data)

            # flip axis
            T = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]], dtype=V.dtype)
            V = np.dot(V, np.transpose(T))

            # scale to the image
            h = self.image_actor.GetMaxYBound() + 1
            model_height = np.amax(V[:,1]) - np.amin(V[:,1])
            scale = (h / model_height) * 0.4
            V *= scale

            poly_data = numpy_to_vtkPolyData(V, cells)

            # save reference to the vertex array
            self._V = V

        # save view of points into the current poly data and the original
        # vertices
        V, cells = vtkPolyData_to_numpy(poly_data)
        self.V0 = V.copy()
        self.V0_prime = V.copy()
        self.V = V

        # setup actor
        self.poly_data = poly_data 
        self.mapper.SetInput(poly_data)
        self.mapper.Modified()
        self.actor.SetVisibility(True)

        # reset selection
        self.last_point = [None, None]
        self.sel_actor.SetVisibility(False)

        # box widget
        self.box_widget.PlaceWidget() 

        self.ren.ResetCamera()
        self.GetRenderWindow().Render()

    def set_transform(self, T):
        print 'Transform:'
        print T

        mat = numpy_to_vtkMatrix4x4(T)
        self.box_transform.SetMatrix(mat)
        self.box_widget.SetTransform(self.box_transform)

        self._transform_update()
            
    def set_correspondences(self, C):
        self.C = C

        proj_lines = self.proj_poly_data.GetLines()
        proj_lines.Reset()

        proj_points = self.proj_poly_data.GetPoints()
        proj_points.SetNumberOfPoints(2*len(C))

        for j in xrange(len(C)):
            (x, y), i = C[j]

            proj_lines.InsertNextCell(2)

            proj_points.SetPoint(2*j, x, y, 0)

            proj_lines.InsertCellPoint(2*j)
            proj_lines.InsertCellPoint(2*j + 1)

        self._projection_update()

    def set_aux_polydata(self, aux_poly_data):
        if aux_poly_data is None:
            self.aux_actor.SetVisibility(False)
            self.actor.SetVisibility(True)
            
            self._projection_update(project_to_aux=False)
        else:
            self.aux_poly_data = aux_poly_data
            self.aux_mapper.SetInput(aux_poly_data)
            self.aux_mapper.Modified()
            self.aux_actor.SetVisibility(True)
            self.actor.SetVisibility(False)

            self._projection_update(project_to_aux=True)

