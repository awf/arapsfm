# select_flexible_regions.py

# Imports
import vtk
from PyQt4 import QtCore, QtGui
from PyQt4_.vtk.QVTKWidget import QVTKWidget
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtk_ import numpy_to_vtkPolyData, iter_vtkCellArray

import os
import numpy as np

from mesh.box_model import box_model
from mesh import faces

from pprint import pprint
from matplotlib import cm

from itertools import count

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
COLORMAP = cm.Set1_r(np.linspace(0., 1., 20))

def make_LUT():
    LUT = vtk.vtkLookupTable()
    LUT.SetNumberOfTableValues(COLORMAP.shape[0])
    LUT.Build()
    for i, c in enumerate(COLORMAP):
        LUT.SetTableValue(i, *c)

    return LUT

LUT = make_LUT()

def make_LABELS():
    n = COLORMAP.shape[0]
    LABEL_TYPES = np.zeros(n, dtype=np.int32)
    LABEL_TYPES[1] = -2
    LABEL_TYPES[2:n/2] = -1
    LABEL_TYPES[n/2:n] = 1

    LABELS = map(str, xrange(n))
    LABELS[0] = LABELS[0] + ' [Fixed]'
    LABELS[1] = LABELS[1] + ' [Independent]'

    m = (n - 2) / 2
    for i in xrange(2, 2 + m):
        LABELS[i] = LABELS[i] + ' [Instance]'
    for i in xrange(2 + m, n):
        LABELS[i] = LABELS[i] + ' [Shared]'

    return LABELS, LABEL_TYPES

LABELS, LABEL_TYPES = make_LABELS()

# MeshView
class MeshView(QVTKWidget):
    def __init__(self, parent=None):
        QVTKWidget.__init__(self, parent)

        self._k = np.array([], dtype=np.int32)

        self.select_only_visible = True
        self.current_label = 0

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
            npy_lookup.flat = self._k.astype(np.float32) / (COLORMAP.shape[0] - 1)

            output.GetPointData().SetScalars(lookup)
        color_points.SetExecuteMethod(callback_color_points)

        source = vtk.vtkCubeSource()

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
        vertices_actor.VisibilityOff()
        ren.AddActor(vertices_actor)

        color_faces = vtk.vtkProgrammableFilter()
        def callback_color_faces():
            input_ = color_faces.GetInput()
            output = color_faces.GetOutput()
            output.ShallowCopy(input_)

            lookup = vtk.vtkFloatArray()
            lookup.SetNumberOfValues(input_.GetNumberOfPolys())
            npy_lookup = vtk_to_numpy(lookup)

            labelled_T = self._k[self.T]
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
        model_actor.VisibilityOff()
        ren.AddActor(model_actor)

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(ren)
        render_window.SetSize(400, 400)
        self.SetRenderWindow(render_window)

        visible = vtk.vtkSelectVisiblePoints()
        visible.SetRenderer(ren)

        picker = vtk.vtkAreaPicker()
        def callback_picker(obj, event):
            props = obj.GetProp3Ds()
            if props.GetNumberOfItems() <= 0:
                return

            extract_geometry = vtk.vtkExtractGeometry()
            extract_geometry.SetImplicitFunction(picker.GetFrustum())
            extract_geometry.SetInput(visible.GetInput())
            extract_geometry.Update()

            unstructured_grid = extract_geometry.GetOutput()
            if unstructured_grid.GetPoints().GetNumberOfPoints() <= 0:
                return

            i = vtk_to_numpy(unstructured_grid.GetPointData().GetArray('i'))

            if self.select_only_visible:
                visible.Update()

                if visible.GetOutput().GetPoints().GetNumberOfPoints() <= 0:
                    return

                i = np.intersect1d(
                    i, vtk_to_numpy(visible.GetOutput().GetPointData().GetArray('i')))

                if i.shape[0] <= 0:
                    return

            self.set_labels(self.current_label, i)

        picker.AddObserver('EndPickEvent', callback_picker)

        iren = self.GetInteractor()
        iren.SetRenderWindow(render_window)
        iren.SetPicker(picker)

        style = vtk.vtkInteractorStyleRubberBandPick()
        style.SetCurrentRenderer(ren)
        iren.SetInteractorStyle(style)

        self.pipeline = locals()

        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                           QtGui.QSizePolicy.Expanding)

    def set_labels(self, label, i=slice(None)):
        self._k[i] = label

        # `color_points` and `color_faces` share input (refer `_setup_pipeline`)
        self.pipeline['color_points'].GetInput().Modified()
        self.update()

    def set_mesh(self, V, T, **kwargs):
        self.V = V
        self.T = T
        self._k = np.zeros(V.shape[0], dtype=np.int32)

        model_poly_data = numpy_to_vtkPolyData(V, faces.faces_to_cell_array(T))

        idFilter = vtk.vtkIdFilter()
        idFilter.SetInput(model_poly_data)
        idFilter.SetIdsArrayName('i')
        idFilter.Update()

        labelled_poly_data = idFilter.GetOutput()

        self.pipeline['color_points'].SetInput(model_poly_data)
        self.pipeline['color_faces'].SetInput(model_poly_data)
        self.pipeline['visible'].SetInput(labelled_poly_data)

        enorm = lambda V: np.sqrt(np.sum(V*V, axis=-1))
        edge_lengths = np.r_[enorm(V[T[:,0]] - V[T[:,1]]),
                             enorm(V[T[:,1]] - V[T[:,2]]),
                             enorm(V[T[:,2]] - V[T[:,0]])]

        length = 0.2 * np.mean(edge_lengths)
        self.pipeline['source'].SetXLength(length)
        self.pipeline['source'].SetYLength(length)
        self.pipeline['source'].SetZLength(length)

        self.pipeline['vertices_actor'].VisibilityOn()
        self.pipeline['model_actor'].VisibilityOn()

        self.pipeline['ren'].ResetCamera()

        self.update()

    def camera_actions(self, *args):
        camera = self.pipeline['ren'].GetActiveCamera()

        for method, method_args in args:
            method = getattr(camera, method)
            method(*method_args)

        self.pipeline['ren'].ResetCamera()
        self.update()

# MainWindow
class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self,parent)
        self._setup_ui()

    def _setup_ui(self):
        view = MeshView()

        select_label = QtGui.QComboBox()
        for label in LABELS:
            select_label.addItem(label)

        def select_label_currentIndexChanged(i):
            view.current_label = i
        select_label.currentIndexChanged.connect(
            select_label_currentIndexChanged)

        reset_all = QtGui.QPushButton('&Reset All')
        def reset_all_clicked():
            label = select_label.currentIndex()
            button = QtGui.QMessageBox.question(
                reset_all.parent(), 
                'Continue?', 
                'Reset all labels to %d?' % label,
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
                QtGui.QMessageBox.No)

            if button != QtGui.QMessageBox.Yes:
                return

            view.set_labels(label)
        reset_all.clicked.connect(reset_all_clicked)

        load_mesh = QtGui.QPushButton('Load &Mesh')
        def load_mesh_clicked():
            filename = QtGui.QFileDialog.getOpenFileName(self, 
                'Load Mesh',
                QtCore.QDir.current().absolutePath(),
                '*.npz *.dat')

            if filename.isEmpty():
                return

            filename = str(filename)
            self.setWindowTitle(filename)
            z = np.load(filename)
            V = z['points']
            T = np.asarray(list(iter_vtkCellArray(z['cells'])),
                           dtype=np.int32)

            view.set_mesh(V, T) 
        load_mesh.clicked.connect(load_mesh_clicked)

        select_only_visible = QtGui.QCheckBox('Select only visible?')
        select_only_visible.setCheckState(QtCore.Qt.Checked)
        def select_only_visible_stateChanged(state):
            view.select_only_visible = (state == QtCore.Qt.Checked)
        select_only_visible.stateChanged.connect(
            select_only_visible_stateChanged)

        view_settings = [('X', ((0., 0., 1.), (1., 0., 0.))),
                         ('Y', ((0., 0., 1.), (0., 1., 0.))),
                         ('Z', ((0., 1., 0.), (0., 0., 1.)))]

        set_view = QtGui.QComboBox()
        for key, settings in view_settings:
            set_view.addItem(key)
        def set_view_activated(i):
            key, settings = view_settings[i]
            view.camera_actions(('SetViewUp', settings[0]),
                                ('SetPosition', settings[1]),
                                ('SetFocalPoint', (0., 0., 0.)))
        set_view.activated.connect(set_view_activated)

        parallel_projection = QtGui.QCheckBox('Parallel projection?')
        def parallel_projection_stateChanged(state):
            view.camera_actions(('SetParallelProjection', 
                                (state == QtCore.Qt.Checked,)))
        parallel_projection.stateChanged.connect(
            parallel_projection_stateChanged)

        save_flexibile = QtGui.QPushButton('&Save Settings')
        def save_flexibile_clicked():
            filename = QtGui.QFileDialog.getSaveFileName(self, 'Save Selection',
                QtCore.QDir.current().absolutePath(),
                '*.npz')

            if filename.isEmpty():
                return

            filename = str(filename)

            # K[i, 0] == 0: "Fixed"
            # K[i, 0] == -1: "Independent"
            #   > K[i, 1] == m: Index of rotation
            # K[i, 0] > 0: "Basis"
            #   > K[i, 1] == n: Index of "Basis" rotation
            K = np.zeros((view._k.shape[0], 2), dtype=np.int32)

            # Independent instance rotations
            i = np.argwhere(view._k == 1).ravel()
            m = i.shape[0]

            K[i, 0] = -1
            K[i, 1] = xrange(m)

            # Shared instance rotations and shared basis rotations
            shared_instance = count(m)
            shared_basis = count(0)
            coefficients = count(1)

            for k in xrange(2, LABEL_TYPES.shape[0]):
                i = np.argwhere(view._k == k).ravel()
                m = i.shape[0]
                if m <= 0:
                    continue

                if LABEL_TYPES[k] == -1:
                    K[i, 0] = -1
                    K[i, 1] = next(shared_instance)
                elif LABEL_TYPES[k] == 1:
                    K[i, 0] = [next(coefficients) for j in xrange(m)]
                    K[i, 1] = next(shared_basis)
                else:
                    raise ValueError('LABEL_TYPES[%d] = %d' % (k, LABEL_TYPES[k]))
            np.savez_compressed(filename, K=K, k=view._k, LABEL_TYPES=LABEL_TYPES)

        save_flexibile.clicked.connect(save_flexibile_clicked)

        load_flexible = QtGui.QPushButton('&Load Settings')
        def load_flexible_clicked():
            filename = QtGui.QFileDialog.getOpenFileName(self, 
                'Load Settings',
                QtCore.QDir.current().absolutePath(),
                '*.npz')

            if filename.isEmpty():
                return

            filename = str(filename)

            z = np.load(filename)
            if not np.all(LABEL_TYPES == z['LABEL_TYPES']):
                QtGui.QMessageBox.warning(self, 
                    'Incompatible `LABEL_TYPES`',
                    'Incompatible `LABEL_TYPES`')

                return

            view.set_labels(z['k'])

        load_flexible.clicked.connect(load_flexible_clicked)

        control_layout = QtGui.QVBoxLayout()
        control_layout.addWidget(set_view)
        control_layout.addWidget(parallel_projection)
        control_layout.addWidget(select_label)
        control_layout.addWidget(reset_all)
        control_layout.addWidget(select_only_visible)
        control_layout.addWidget(load_mesh)
        control_layout.addWidget(load_flexible)
        control_layout.addWidget(save_flexibile)
        control_layout.addStretch(1)

        main_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(control_layout, stretch=0)
        main_layout.addWidget(view, stretch=1)

        main_widget = QtGui.QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.ui = locals()
        
# main
def main():
    V, T = box_model(5, 15, 1.0, 1.0) 

    qapp = QtGui.QApplication([])

    window = MainWindow()
    window.show()

    # view = MeshView()
    # view.set_mesh(V, T)
    # view.select_only_visible = True
    # view.current_label = 1
    # view.camera_actions(('SetViewUp', (0., 0., 1.)), 
    #                     ('SetPosition', (-1.0, 0., 0.)),
    #                     ('SetFocalPoint', (0., 0., 0.)))
    # view.show()

    qapp.exec_()

if __name__ == '__main__':
    # test_vtkInteractorStyleRubberBandPick()
    main()

