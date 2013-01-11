# interactive_core_recovery.py

# Imports
import numpy as np
from vtk_ import *
from PyQt4 import QtGui, QtCore
from PyQt4_.vtk.QVTKWidget import QVTKWidget
from misc.bunch import Bunch

from mesh import geometry

from core_recovery2.problem import CoreRecoverySolver

# Functions

# int2dbl
def int2dbl(*x):
    return np.array(x, dtype=np.float64) / 255. 

# updates
def updates(*keys):
    keys = list(keys)

    def fn_wrapper(fn):
        def wrapped_fn(self, *args, **kwargs):
            fn(self, *args, **kwargs)
            
        return wrapped_fn
    return fn_wrapper

# InteractiveMeshView
class InteractiveMeshView(QVTKWidget):
    vertexSelected = QtCore.pyqtSignal(name='vertexSelected')

    def __init__(self, parent=None):
        QVTKWidget.__init__(self, parent)
        self._setup_renderer()

    def _setup_renderer(self):
        self._actors = Bunch()
        self._data = Bunch()

        ren = vtk.vtkRenderer()
        ren.SetBackground(1., 1., 1.)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(ren)

        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.01)
        picker.AddObserver('EndPickEvent', self._observe_pick)

        iren = self.GetInteractor()
        iren.SetPicker(picker)

        style = vtk.vtkInteractorStyleImage()
        style.SetInteractionModeToImage3D()
        style.SetCurrentRenderer(ren)
        iren.SetInteractorStyle(style)

        _p = locals()
        del _p['self']
        self._p = Bunch(_p)

        self._add_image()
        self._add_mesh()
        self._add_constraints((255, 0, 0), 'user', 5.0)
        self._add_constraints((255, 255, 0), 'silhouette', 3.0)

    def _add_image(self):
        reader = vtk.vtkPNGReader()

        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputConnection(reader.GetOutputPort())

        self._p.ren.AddActor(image_actor)
        self._actors.image = image_actor
        self._reader = reader

    def _add_mesh(self):
        mapper = vtk.vtkPolyDataMapper()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*int2dbl(31, 120, 180))
        actor.GetProperty().SetLighting(True)

        box_widget = vtk.vtkBoxWidget()
        box_widget.SetPlaceFactor(1.00)
        box_widget.SetInteractor(self._p.iren)
        box_widget.SetProp3D(actor)

        def transform_actor(obj, event):
            transform = vtk.vtkTransform()
            obj.GetTransform(transform)

            mat = self.box_transform.GetMatrix()
            if mat is None:
                T = np.c_[np.eye(3), (0.,0.,0.)]
            else:
                T = vtkMatrix4x4_to_numpy(mat)

            self.set_geometry(np.dot(self.V0, np.transpose(T[:3,:3])) + 
                              T[:3,-1])

        box_widget.AddObserver('InteractionEvent', transform_actor)

        self._p.box_widget = box_widget
        self._p.ren.AddActor(actor)
        self._actors.mesh = actor

    def _add_constraints(self, color, attr_name, radius): 
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(0)
        lines = vtk.vtkCellArray()

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)

        tube = vtk.vtkTubeFilter()
        tube.SetInput(poly_data)
        tube.SetRadius(radius)
        tube.SetNumberOfSides(8)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*int2dbl(color))
        actor.GetProperty().SetLighting(True)
        actor.PickableOff()

        self._p.ren.AddActor(actor)
        setattr(self._actors, attr_name, actor)
        setattr(self._data, attr_name, poly_data)

    def set_mesh(self, T):
        self.T = T
        cells = faces_to_vtkCellArray(T)

        V = np.zeros((T.shape[0], 3), dtype=np.float32)
        self._data.mesh = numpy_to_vtkPolyData(V, cells)
        self._actors.mesh.GetMapper().SetInput(self._data.mesh)

        self._data.V = vtkPolyData_to_numpy(self._data.mesh)[0]

    def set_image(self, filename):
        self._reader.SetFileName(filename)

    def set_user_constraints(self, C, P):
        n = C.shape[0]

        lines = self._data.user.GetLines()
        lines.Reset()

        points = self._data.user.GetPoints()
        points.SetNumberOfPoints(2*n)

        for i in xrange(n):
            points.SetPoint(2*i, P[i,0], P[i,1], 0.)

            lines.InsertNextCell(2*i)
            lines.InsertCellPoint(2*i + 1)

        self.C = C
        self.P = P

    def set_silhouette_projection(self, S):
        n = S.shape[0]

        lines = self._data.silhouette.GetLines()
        lines.Reset()

        points = self._data.silhouette.GetPoints()
        points.SetNumberOfPoints(2*n)

        for i in xrange(n):
            points.SetPoint(2*i, S[i,0], S[i,1], 0)

            lines.InsertNextCell(2*i)
            lines.InsertCellPoint(2*i + 1)

        self.S = S

    def set_initial_geometry(self, V0, **kwargs):
        # pass `no_update`
        self.V0 = V0.copy()
        self.set_geometry(V0, **kwargs)
        self._p.box_widget.PlaceWidget()

    @updates('user_constraints', 'silhouette_projection')
    def set_geometry(self, V):
        V_flat = np.require(V, requirements='C', dtype=np.float32).flat
        self._data.V.flat = V_flat
        self._data.mesh.Modified()

    @updates('silhouette_projection')
    def set_silhouette_preimage(self, L, U):
        self.L = L
        self.U = U

    def _update_user_constraints(self):
        user_points = self._data.user.GetPoints()
        for i in xrange(self.C.shape[0]):
            user_points.SetPoint(2*i + 1, *self._data.V[self.C[i]])
        self._data.user.Modified()

    def _update_silhouette_projection(self):
        silhouette_points = self._data.silhouette.GetPoints()
        Q = geometry.path2pos(self.V, self.T, self.L, self.U)
        for i in xrange(self.S.shape[0]):
            silhouette_points.SetPoint(2*i + 1, *Q[i])
        self._data.silhouette.Modified()

    def update(self):
        self.GetRenderWindow().Render()

    def highlight_silhouette_points(self, i):
        # TODO highlight the current silhouette positions
        pass

    def _observe_pick(self, obj):
        # TODO highlight the current selected vertex

        # TODO emit vertexSelected
        pass

# MainWindow
class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self._setup_ui()

    def _setup_ui(self):
        self.mesh_view = InteractiveMeshView()

        main_layout = QtGui.QGridLayout()
        main_layout.addWidget(self.mesh_view)

        main_widget = QtGui.QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

# main
def main():
    qapp = QtGui.QApplication([])
    main_window = MainWindow()

    solver = np.load('solve_core_recovery_test/Impala_1_Impala_1_85,86,87,88_1,1,1024,0.5,2,0,0,0,0,0,0,0_1,1,1,32,1/solver.dat')
    m = main_window.mesh_view
    m.set_mesh(solver.T)
    m.set_image(solver.frames[0])
    m.set_user_constraints(solver.C[0], solver.P[0])
    m.set_silhouette_projection(solver.S[0])
    m.set_initial_geometry(solver._s.V1[0])
    
    main_window.show()

    qapp.exec_()

if __name__ == '__main__':
    main()

