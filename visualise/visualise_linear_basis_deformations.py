# visualise_linear_basis_deformations.py

# Imports
import os, argparse
import numpy as np
from vtk_ import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from PyQt4_.vtk.QVTKWidget import QVTKWidget
from misc.pickle_ import dump, load
from misc.bunch import Bunch

from scipy import linalg

from operator import add

# int2dbl
def int2dbl(*x):
    return np.array(x, dtype=np.float64) / 255. 

# QuickMeshView
class QuickMeshView(QVTKWidget):
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

        style = InteractorStyle()
        style.SetMagnification(2.)
        style.SetCurrentRenderer(ren)

        iren = self.GetInteractor()
        iren.SetInteractorStyle(style)

        _p = locals()
        del _p['self']
        self._p = Bunch(_p)

        self._add_mesh()
        self._add_image()

    def _add_image(self):
        reader = vtk.vtkPNGReader()

        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputConnection(reader.GetOutputPort())
        image_actor.VisibilityOff()

        self._p.ren.AddActor(image_actor)
        self._actors.image = image_actor
        self._reader = reader
        
    def _add_mesh(self):
        self._p.loop = vtk.vtkLoopSubdivisionFilter()

        self._p.normals = vtk.vtkPolyDataNormals()
        self._p.normals.SetInputConnection(self._p.loop.GetOutputPort())
        self._p.normals.ComputeCellNormalsOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._p.normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLighting(True)
        actor.GetProperty().SetColor(*int2dbl(179, 179, 179))

        self._p.ren.AddActor(actor)
        self._actors.mesh = actor

    def set_image(self, filename):
        self._reader.SetFileName(filename)
        self._actors.image.VisibilityOn()

    def set_mesh(self, T, V):
        self.T = T
        self.V = V

        cells = faces_to_vtkCellArray(T)
        self._data.mesh = numpy_to_vtkPolyData(V, cells)

        self._p.loop.SetInput(self._data.mesh)
        self._data.mesh.Modified()

    def refresh(self):
        self.GetRenderWindow().Render()

    def reset_camera(self):
        camera = self._p.ren.GetActiveCamera()
        camera.SetParallelProjection(True)
        
        self._p.ren.ResetCamera()

    def sizeHint(self):
        return QSize(960, 540)

# MainWindow
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.frames = None
        self._setup_ui()

    def _setup_ui(self):
        self.mesh_view = QuickMeshView()
        self.mesh_view.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)

        view_layout = QVBoxLayout()
        view_layout.addWidget(self.mesh_view)

        self.instance_slider = QSlider(Qt.Horizontal)
        self.instance_slider.setMinimum(0)
        self.instance_slider.setTracking(True)
        self.instance_slider.valueChanged.connect(self.set_instance)

        self.basis_slider_layout = QHBoxLayout()

        zero_button = QPushButton('&Zero')
        zero_button.clicked.connect(self._zero_modes)
        button_layout = QHBoxLayout()
        button_layout.addWidget(zero_button)
        button_layout.addStretch(1)

        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(self.instance_slider)
        ctrl_layout.addLayout(self.basis_slider_layout)
        ctrl_layout.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(view_layout, stretch=1)
        main_layout.addLayout(ctrl_layout)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def slider_changed(self, unused_value):
        self.set_instance(set_basis_shapes=False)

    def _zero_modes(self):
        self._set_basis_states(np.zeros(self.L.shape[1]))
        self.set_instance(set_basis_shapes=False)

    def _set_basis_states(self, l):
        for i, slider in enumerate(self.basis_sliders):
            j = np.argmin(np.abs(l[i] - self.basis_slider_values[i]))
            slider.setValue(j)

    def _get_shape(self):
        l = np.empty(self.L.shape[1], dtype=np.float64)
        for i, slider in enumerate(self.basis_sliders):
            slider.blockSignals(True)
            l[i] = self.basis_slider_values[i][slider.value()]
            slider.blockSignals(False)
        return self.V0 + np.sum(l.reshape(-1, 1, 1) * self.Vb, axis=0)

    def set_instance(self, i=None, set_basis_shapes=True):
        if i is None:
            i = self.instance_slider.value()

        if set_basis_shapes:
            self._set_basis_states(self.L[i])

        V = self._get_shape()

        self.mesh_view.set_mesh(self.T, V)
        if self.frames is not None:
            self.mesh_view.set_image(self.frames[i])

        self.mesh_view.refresh()
        
    def set_shape_space(self, T, V0, Vb, L):
        self.T = T
        self.V0 = V0
        self.Vb = Vb
        self.L = L

        k = Vb.shape[0]
        min_, max_ = np.amin(L, axis=0), np.amax(L, axis=0)
        range_ = max_ - min_

        n = 2
        num_positions = 100
        slider_centre = 0.5 * (max_ + min_)
        slider_min = slider_centre - n * 0.5 * range_
        slider_max = slider_centre + n * 0.5 * range_

        self.basis_sliders = []
        self.basis_slider_values = []

        for i in xrange(k):
            slider_values = np.linspace(slider_min[i], 
                                        slider_max[i], 
                                        num_positions,
                                        endpoint=True)
            self.basis_slider_values.append(slider_values)

            slider = QSlider(Qt.Vertical)
            slider.setMinimum(0)
            slider.setMaximum(num_positions - 1)
            slider.sliderMoved.connect(self.slider_changed)
            self.basis_sliders.append(slider)
            self.basis_slider_layout.addWidget(slider)

        self.set_instance(0)
        self.mesh_view.reset_camera()

    def set_frames(self, frames):
        self.frames = frames

# decompose_basis_shapes
def decompose_basis_shapes(states, k=None, explained_variance=0.95):
    V = map(lambda s: s['V'].ravel(), states)
    W = np.vstack(V)
    W0 = np.mean(W, axis=0)
    W -= W0

    U, s, Vt = linalg.svd(W, full_matrices=False)

    if k is None:
        cum_s2 = np.cumsum(s*s)
        cum_s2 /= cum_s2[-1]

        required_k = np.argwhere(cum_s2 >= explained_variance).ravel()
        k = required_k[0]

    V0 = W0.reshape(-1, 3)
    Vb = Vt[:k].reshape(k, -1, 3)
    L = np.dot(U[:,:k], np.diag(s[:k]))

    return states[0]['T'], V0, Vb, L


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('states_path')
    parser.add_argument('-k', type=int)
    parser.add_argument('--show_frames', 
                        default=False,
                        action='store_true')

    args = parser.parse_args()

    z = np.load(args.states_path)
    states = z['states']
    frames = map(lambda s: s['image'], states)
    T, V0, Vb, L = decompose_basis_shapes(states, k=args.k)

    global qapp
    qapp = QApplication([])
    main_window = MainWindow()

    if args.show_frames:
        main_window.set_frames(frames)

    main_window.set_shape_space(T, V0, Vb, L)

    main_window.show()

    qapp.exec_()

if __name__ == '__main__':
    main()

