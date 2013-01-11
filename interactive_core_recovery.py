# interactive_core_recovery.py

# Imports
import os, argparse
import numpy as np
from vtk_ import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4_.vtk.QVTKWidget import QVTKWidget
from misc.bunch import Bunch

import mesh.geometry
import mesh.faces

from core_recovery2.problem import CoreRecoverySolver

from misc.pickle_ import dump, load

# Constants
WORKING_SOLVER_OPTIONS = dict(maxIterations=50, 
                              gradientThreshold=1e-5,
                              updateThreshold=1e-5,
                              improvementThreshold=1e-5,
                              verbosenessLevel=1)

WORKING_CONFIGURATION = dict(max_restarts=3)
PIXEL_SELECTION_THRESHOLD = 100.

# Functions

# int2dbl
def int2dbl(*x):
    return np.array(x, dtype=np.float64) / 255. 

# updates
def updates(*keys):
    def fn_wrapper(fn):
        def wrapped_fn(self, *args, **kwargs):
            update = kwargs.pop('update', True)
            fn(self, *args, **kwargs)

            if update:
                for name in map(lambda k: '_update_%s' % k, keys):
                    method = getattr(self, name)
                    method()

        return wrapped_fn
    return fn_wrapper

# InteractiveMeshView
class InteractiveMeshView(QVTKWidget):
    # vertexSelected = pyqtSignal(name='vertexSelected')
    # silhouettePointSelected = pyqtSignal(name='silhouettePointSelected')
    transformed = pyqtSignal(name='transformed')

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
        self.SetRenderWindow(render_window)

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
        self._add_constraints((227, 26, 28), 'user', 4.0)
        self._add_constraints((255, 255, 51), 'silhouette', 2.0)
        self._add_points((255, 255, 51), 'silhouette_points', 5.0)
        self._add_points((255, 127, 0), 'preimage_points', 5.0)

    def _add_image(self):
        reader = vtk.vtkPNGReader()

        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputConnection(reader.GetOutputPort())

        self._p.ren.AddActor(image_actor)
        self._actors.image = image_actor
        self._reader = reader

    def _add_mesh(self):
        color_face = vtk.vtkProgrammableFilter()

        def color_face_callback():
            input_ = color_face.GetInput()
            output = color_face.GetOutput()
            output.ShallowCopy(input_)

            color_lookup = vtk.vtkIntArray()
            color_lookup.SetNumberOfValues(input_.GetNumberOfPolys())
            npy_cl = vtk_to_numpy(color_lookup)
            npy_cl.fill(0)

            try:
                npy_cl[self.L] = 1
            except AttributeError:
                pass

            output.GetCellData().SetScalars(color_lookup)
            
        color_face.SetExecuteMethod(color_face_callback)

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(2)
        lut.Build()

        lut_SetTableValue = lut.SetTableValue
        def SetTableValue(i, r, g, b, alpha=1.0):
            return lut_SetTableValue(i, r, g, b, alpha)
        lut.SetTableValue = SetTableValue
        lut.SetTableValue(0, *int2dbl(179, 179, 179))
        lut.SetTableValue(1, *int2dbl(255, 127, 0))

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(color_face.GetOutputPort())
        mapper.SetScalarRange(0, 1)
        mapper.SetLookupTable(lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLighting(True)

        box_widget = vtk.vtkBoxWidget()
        box_widget.SetPlaceFactor(1.00)
        box_widget.SetInteractor(self._p.iren)
        box_widget.SetProp3D(actor)

        def transform_actor(obj, event):
            transform = vtk.vtkTransform()
            obj.GetTransform(transform)

            mat = transform.GetMatrix()
            if mat is None:
                T = np.c_[np.eye(3), (0.,0.,0.)]
            else:
                T = vtkMatrix4x4_to_numpy(mat)

            self.set_geometry(np.dot(self.V0, np.transpose(T[:3,:3])) + 
                              T[:3,-1])

            self.transformed.emit()

        box_widget.AddObserver('InteractionEvent', transform_actor)

        self._p.box_widget = box_widget
        self._p.ren.AddActor(actor)
        self._p.color_mesh = color_face
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
        tube.SetNumberOfSides(16)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*int2dbl(color))
        actor.PickableOff()

        self._p.ren.AddActor(actor)
        setattr(self._actors, attr_name, actor)
        setattr(self._data, attr_name, poly_data)

    def _add_points(self, color, attr_name, radius):
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(0)
        
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInput(poly_data)
        glyph.SetSourceConnection(sphere.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOff()
        actor.GetProperty().SetColor(*int2dbl(color))
        
        self._p.ren.AddActor(actor)
        setattr(self._actors, attr_name, actor)
        setattr(self._data, attr_name, poly_data)

    def set_mesh(self, N, T):
        self.T = T
        cells = faces_to_vtkCellArray(T)

        V = np.zeros((N, 3), dtype=np.float32)
        self._data.mesh = numpy_to_vtkPolyData(V, cells)
        self._p.color_mesh.SetInput(self._data.mesh)

        self.V = vtkPolyData_to_numpy(self._data.mesh)[0]
        self._selected_V = - 1

    def set_image(self, filename):
        self._reader.SetFileName(filename)

    def _set_constraints(self, attr_name, P):
        data = getattr(self._data, attr_name)
        lines = data.GetLines()
        lines.Reset()

        n = P.shape[0]
        points = data.GetPoints()
        points.SetNumberOfPoints(2*n)

        for i in xrange(n):
            points.SetPoint(2*i, P[i,0], P[i,1], 0.)

            lines.InsertNextCell(2)
            lines.InsertCellPoint(2*i)
            lines.InsertCellPoint(2*i + 1)

    def set_user_constraints(self, C, P):
        self.C, self.P = C, P
        self._set_constraints('user', P)

    @updates('silhouette_points')
    def set_silhouette_constraints(self, S):
        self.S = S
        self._selected_S = np.zeros(S.shape[0], dtype=bool)
        self._set_constraints('silhouette', S)

    def set_initial_geometry(self, V0, update=False):
        self.V0 = V0.copy()
        self.set_geometry(V0, update=update)
        self._p.box_widget.PlaceWidget()

    @updates('mesh')
    def set_geometry(self, V):
        V_flat = np.require(V, requirements='C', dtype=np.float32).flat
        self.V.flat = V_flat

    @updates('mesh')
    def set_silhouette_preimage(self, L, U):
        self.L, self.U = L.copy(), U.copy()

    @updates('silhouette_projection', 'user_constraints', 'preimage_points')
    def _update_mesh(self):
        self._data.mesh.Modified()

    @updates('silhouette_points')
    def clear_selection(self):
        self._selected_S.fill(False)

    def reset_camera(self):
        camera = self._p.ren.GetActiveCamera()
        camera.SetParallelProjection(True)
        
        self._p.ren.ResetCamera()

    def refresh(self):
        self.GetRenderWindow().Render()

    def _update_user_constraints(self):
        user_points = self._data.user.GetPoints()
        for i in xrange(self.C.shape[0]):
            user_points.SetPoint(2*i + 1, self.V[self.C[i]])
        self._data.user.Modified()

    def _update_silhouette_projection(self):
        silhouette_points = self._data.silhouette.GetPoints()
        Q = mesh.geometry.path2pos(self.V, self.T, self.L, self.U)
        for i in xrange(self.S.shape[0]):
            silhouette_points.SetPoint(2*i + 1, *Q[i])
        self._data.silhouette.Modified()

    def _update_silhouette_points(self):
        selected_points = self._data.silhouette_points.GetPoints()
        I = np.argwhere(self._selected_S).ravel()
        selected_points.SetNumberOfPoints(I.shape[0])
        for j, i in enumerate(I):
            selected_points.SetPoint(j, np.r_[self.S[i], 0.])
        self._data.silhouette_points.Modified()

    def _update_preimage_points(self):
        preimage_points = self._data.preimage_points.GetPoints()
        i = self._selected_V
        if i < 0:
            preimage_points.SetNumberOfPoints(0)
        else:
            preimage_points.SetNumberOfPoints(1)
            preimage_points.SetPoint(0, self.V[i])

        self._data.preimage_points.Modified()

    def _observe_pick(self, obj, event):
        i = self._p.picker.GetPointId()

        if i < 0:
            return

        prop = self._p.picker.GetProp3D()

        if prop is self._actors.image:
            y, x = divmod(i, self._actors.image.GetMaxXBound() + 1)
            r = self.S - (x, y)

            D = np.sum(r * r, axis=1)
            i = np.argmin(D)

            if np.sqrt(D[i]) >= PIXEL_SELECTION_THRESHOLD:
                return

            self._selected_S[i] ^= True

            return self._update_silhouette_points()

        elif prop is self._actors.mesh:
            self._selected_V = i

            return self._update_preimage_points()
            
# MainWindow
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self._setup_ui()

        self._update_count = 0

    def _setup_ui(self):
        control_layout = QVBoxLayout()

        instance_layout = QHBoxLayout()
        self.instance_label = QLabel('I&ndex:')
        instance_layout.addWidget(self.instance_label)

        self.instance_slider = QSlider(Qt.Horizontal)
        self.instance_slider.setMinimum(0)
        self.instance_slider.setTracking(True)
        self.instance_slider.valueChanged.connect(self.set_instance)
        instance_layout.addWidget(self.instance_slider)
        self.instance_label.setBuddy(self.instance_slider)

        control_layout.addLayout(instance_layout)

        self.enable_silhouette = QCheckBox('&Enable Silhouette')
        control_layout.addWidget(self.enable_silhouette)

        self.fixed_scale = QCheckBox('&Fixed Scale')
        self.fixed_scale.setChecked(True)
        control_layout.addWidget(self.fixed_scale)

        self.fixed_global_rotation = QCheckBox('Fixed &Global Rotation')
        self.fixed_global_rotation.setChecked(True)
        control_layout.addWidget(self.fixed_global_rotation)

        self.refresh = QCheckBox('&Refresh')
        self.refresh.setChecked(True)
        control_layout.addWidget(self.refresh)

        self.solve_instance_button = QPushButton('Solve &Instance')
        self.solve_instance_button.clicked.connect(self.solve_instance)
        control_layout.addWidget(self.solve_instance_button)

        self.solve_silhouette_button = QPushButton('Solve Si&lhouette')
        self.solve_silhouette_button.clicked.connect(self.solve_silhouette)
        control_layout.addWidget(self.solve_silhouette_button)

        self.update_from_selection_button = QPushButton('&Update from Selection')
        self.update_from_selection_button.clicked.connect(
            self.update_from_selection)
        control_layout.addWidget(self.update_from_selection_button)

        self.clear_selection_button = QPushButton('&Clear Selection')
        self.clear_selection_button.clicked.connect(self.clear_selection)
        control_layout.addWidget(self.clear_selection_button)

        def control_layout_add_separator():
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            control_layout.addWidget(line)

        control_layout_add_separator()

        self.save_button = QPushButton('&Save')
        self.save_button.clicked.connect(self.save_solver)
        control_layout.addWidget(self.save_button)

        control_layout.addStretch(1)

        self.mesh_view = InteractiveMeshView()
        self.mesh_view.transformed.connect(self.update_geometry)
        self.mesh_view.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)

        view_layout = QVBoxLayout()
        view_layout.addWidget(self.mesh_view)

        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addLayout(view_layout, stretch=1)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def set_instance(self, i):
        self.instance_label.setText(self.instance_format_string % i)
        self.mesh_view.set_image(self.solver.frames[i])
        self.mesh_view.set_mesh(self.solver._s.V.shape[0], self.solver.T)
        self.mesh_view.set_user_constraints(self.solver.C[i], self.solver.P[i])
        self.mesh_view.set_silhouette_constraints(self.solver.S[i])
        self.mesh_view.set_silhouette_preimage(self.solver._s.L[i], self.solver._s.U[i], update=False)
        self.mesh_view.set_initial_geometry(self.solver._s.V1[i], update=True)
        self.mesh_view.refresh()

    def solve_instance(self):
        i = self.instance_slider.value()
        enable_silhouette = self.enable_silhouette.isChecked()

        def update_instance(iteration, computeDerivatives):
            if not computeDerivatives:
                return

            if enable_silhouette:
                self.mesh_view.set_silhouette_preimage(self.solver._s.L[i],
                                                       self.solver._s.U[i],
                                                       update=False)

            self.mesh_view.set_initial_geometry(self.solver._s.V1[i], update=True)

            self.mesh_view.refresh()

        if self.refresh.isChecked():
            callback = update_instance
        else:
            callback = None

        self.solver.solve_instance(
            i, 
            fixed_global_rotation=self.fixed_global_rotation.isChecked(),
            fixed_scale=self.fixed_scale.isChecked(),
            no_silhouette=not enable_silhouette,
            callback=callback)

    def solve_silhouette(self):
        i = self.instance_slider.value()
        self.solver.solve_silhouette(i)

        self.mesh_view.set_silhouette_preimage(self.solver._s.L[i],
                                               self.solver._s.U[i],
                                               update=True)
        self.mesh_view.refresh()

    def update_geometry(self):
        i = self.instance_slider.value()
        self.solver._s.V1[i].flat = self.mesh_view.V.flat

    def clear_selection(self):
        self.mesh_view.clear_selection()
        self.mesh_view.refresh()

    def update_from_selection(self):
        if self.mesh_view._selected_V < 0:
            return

        I = np.argwhere(self.mesh_view._selected_S).ravel()
        if I.shape[0] <= 0:
            return

        u = None

        for l, face in enumerate(self.solver.T):
            for j, v in enumerate(face):
                if v == self.mesh_view._selected_V:
                    if j == 0:
                        u = np.r_[1., 0.]
                    elif j == 1:
                        u = np.r_[0., 1.]
                    else:
                        u = np.r_[0., 0.]

                    break

            if u is not None:
                break
        else:
            raise ValueError('unable to find vertex: %d' %
                             self.mesh_view._selected_V)

        self._update_count += I.shape[0]
        print '_update_count:', self._update_count

        i = self.instance_slider.value()
        self.solver._s.L[i][I] = l
        self.solver._s.U[i][I] = u
        self.mesh_view.set_silhouette_preimage(self.solver._s.L[i],
                                               self.solver._s.U[i],
                                               update=True)
        self.mesh_view.refresh()

    def load_solver(self, path):
        self.solver_path = path
        self.solver = load(path)

        max_index = len(self.solver.frames) - 1
        n = int(np.floor(np.log10(max_index) + 1))
        self.instance_format_string = 'I&ndex: %{n}d'.format(n=n)

        self.instance_slider.setMaximum(max_index)
        self.instance_slider.setValue(0)
        self.set_instance(0)
        self.mesh_view.reset_camera()

        self.original_solver_options = self.solver.solver_options.copy()
        self.solver.solver_options.update(WORKING_SOLVER_OPTIONS)

        self.original_configuration = {}
        for k, v in WORKING_CONFIGURATION.iteritems():
            self.original_configuration[k] = getattr(self.solver, k)
            setattr(self.solver, k, v)

    def save_solver(self):
        head, tail = os.path.split(self.solver_path)
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save', path, '*.dat') 

        if path.isEmpty():
            return

        pickle_.dump(str(path), self.solver)
        self.solver_path = path

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('solver_path')
    args = parser.parse_args()

    qapp = QApplication([])
    main_window = MainWindow()
    main_window.load_solver(args.solver_path)
    main_window.show()

    qapp.exec_()

if __name__ == '__main__':
    main()
