# user_constraints_ui.py 

# Imports
import numpy as np

from vtk_ import *

from PyQt4 import QtGui, QtCore
from mesh_view import InteractiveMeshView

from matplotlib.pyplot import imread
from operator import itemgetter
from itertools import izip
from pprint import pprint

import os
from os.path import splitext

# Solver
from core_recovery.lm_solvers import solve_single_arap_proj

# GUI

# MainWindow
class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.setup_state()
        self.setup_ui()

    def setup_state(self):  
        self.image_full_path = None
        self.last_image_path = None
        self.last_mesh_path = None
        self.last_correspondences_path = None
        self.arap_poly_data = None

    def setup_ui(self):
        self.load_image_pb = QtGui.QPushButton('Load &Image')
        self.next_image_pb = QtGui.QPushButton('&Next Image')
        self.prev_image_pb = QtGui.QPushButton('&Prev Image')
        self.load_mesh_pb = QtGui.QPushButton('Load &Mesh')
        self.load_pb = QtGui.QPushButton('&Load')
        self.save_pb = QtGui.QPushButton('&Save')
        self.add_pb = QtGui.QPushButton('&Add')
        self.remove_pb = QtGui.QPushButton('&Remove')
        self.toggle_pb = QtGui.QPushButton('&Enable/Disable')
        self.update_pb = QtGui.QPushButton('&Update')
        self.reset_pb = QtGui.QPushButton('Reset')
        self.print_info_pb = QtGui.QPushButton('Print Inf&o')
        self.apply_arap_pb = QtGui.QPushButton('&Deform')
        self.toggle_arap_view_pb = QtGui.QPushButton('To&ggle View')
        self.toggle_arap_view_pb.setCheckable(True)
        self.max_iterations_le = QtGui.QLineEdit('10')

        pb_layout = QtGui.QGridLayout()

        self.items = QtGui.QListWidget()

        font = QtGui.QFont('Monospace', 10, QtGui.QFont.Normal, False)
        self.items.setFont(font)

        ctrl_layout = QtGui.QVBoxLayout()
        def ctrl_layout_add_separator():
            line = QtGui.QFrame()
            line.setFrameShape(QtGui.QFrame.HLine)
            line.setFrameShadow(QtGui.QFrame.Sunken)
            ctrl_layout.addWidget(line)

        ctrl_layout.addWidget(self.items)
        ctrl_layout_add_separator()
        
        item_pbs_layout = QtGui.QGridLayout()
        item_pbs_layout.addWidget(self.add_pb, 0, 0)
        item_pbs_layout.addWidget(self.remove_pb, 0, 1)
        item_pbs_layout.addWidget(self.toggle_pb, 1, 0)
        item_pbs_layout.addWidget(self.update_pb, 1, 1)
        ctrl_layout.addLayout(item_pbs_layout)
        ctrl_layout_add_separator()

        load_pbs_layout = QtGui.QGridLayout()
        load_pbs_layout.addWidget(self.load_image_pb, 0, 0)
        load_pbs_layout.addWidget(self.load_mesh_pb, 0, 1)
        load_pbs_layout.addWidget(self.prev_image_pb, 1, 0)
        load_pbs_layout.addWidget(self.next_image_pb, 1, 1)
        ctrl_layout.addLayout(load_pbs_layout)
        ctrl_layout_add_separator()

        correspondences_pbs_layout = QtGui.QGridLayout()
        correspondences_pbs_layout.addWidget(self.save_pb, 0, 0)
        correspondences_pbs_layout.addWidget(self.load_pb, 0, 1)
        correspondences_pbs_layout.addWidget(self.reset_pb, 1, 0)
        correspondences_pbs_layout.addWidget(self.print_info_pb, 1, 1)
        ctrl_layout.addLayout(correspondences_pbs_layout)
        ctrl_layout_add_separator()

        arap_layout = QtGui.QGridLayout()
        arap_layout.addWidget(self.max_iterations_le, 0, 0)
        arap_layout.addWidget(self.apply_arap_pb, 0, 1)
        arap_layout.addWidget(self.toggle_arap_view_pb, 0, 2)
        ctrl_layout.addLayout(arap_layout)

        self.mesh_view = InteractiveMeshView()

        view_layout = QtGui.QVBoxLayout()
        view_layout.addWidget(self.mesh_view)

        main_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(ctrl_layout)
        main_layout.addLayout(view_layout, stretch=1)

        main_widget = QtGui.QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.max_iterations_le.setSizePolicy(QtGui.QSizePolicy.Minimum,
                                             QtGui.QSizePolicy.Minimum)

        self.load_image_pb.clicked.connect(self.load_image)
        self.prev_image_pb.clicked.connect(self.prev_image)
        self.next_image_pb.clicked.connect(self.next_image)
        self.load_mesh_pb.clicked.connect(self.load_mesh)
        self.add_pb.clicked.connect(self.add_correspondence)
        self.remove_pb.clicked.connect(self.remove_correspondence)
        self.toggle_pb.clicked.connect(self.toggle_correspondence)
        self.update_pb.clicked.connect(self.update_correspondence)
        self.save_pb.clicked.connect(self.save)
        self.load_pb.clicked.connect(self.load)
        self.print_info_pb.clicked.connect(self.print_info)

        self.reset_pb.clicked.connect(self.mesh_view.reset)

        self.apply_arap_pb.clicked.connect(self.apply_arap_deformation)
        self.toggle_arap_view_pb.clicked.connect(self.toggle_arap_view)

        self.items.currentRowChanged.connect(self.update_selection)

        self.window_title_stem = 'CoreRecovery: '
        self.setWindowTitle(self.window_title_stem)

    def load_image(self):
        path = self.last_image_path
        if path is None:
            path = QtCore.QDir.current().absolutePath()

        filename = QtGui.QFileDialog.getOpenFileName(self, 'Load Image',
            path, '*.png')

        if filename.isEmpty():
            return

        self._load_image(str(filename))

    def _load_image(self, full_path):
        self.mesh_view.set_image(full_path)
        self.last_image_path = os.path.split(full_path)[0]
        self.image_full_path = full_path
        self.setWindowTitle(self.window_title_stem + full_path) 

    def _delta_image_number(self, delta):
        if self.image_full_path is None:
            return None

        def number_from_filename(filename):
            root, ext = os.path.splitext(filename)
            try:
                return int(root.split('_')[-1])
            except ValueError:
                return None

        dir_, filename = os.path.split(self.image_full_path)
        n = number_from_filename(filename)
        if n is None:
            return None

        n += delta

        files = os.listdir(dir_)
        numbers = map(number_from_filename, files)
        for i, m in enumerate(numbers):
            if m is not None and m == n:
                return os.path.join(dir_, files[i])
        
        return None
                
    def delta_image_number(self, delta):
        image_path = self._delta_image_number(delta)
        if image_path is None:
            return

        self._load_image(image_path)

    def prev_image(self):
        self.delta_image_number(-1)

    def next_image(self):
        self.delta_image_number(1)

    def load_mesh(self):
        path = self.last_mesh_path
        if path is None:
            path = QtCore.QDir.current().absolutePath()

        filename = QtGui.QFileDialog.getOpenFileName(self, 'Load Mesh',
            path, 'Meshes (*.stl *.npz)')

        if filename.isEmpty():
            return

        self._load_mesh(str(filename))

    def _load_mesh(self, filename):
        root, ext = splitext(filename)

        if ext == '.stl':
            reader = vtk.vtkSTLReader()
            reader.SetFileName(str(filename))
            reader.Update()

            poly_data = reader.GetOutput()

        elif ext == '.npz':
            z = np.load(filename)
            V, T_ = z['points'], z['cells']
            z.close()

            poly_data = numpy_to_vtkPolyData(V, T_)

        self.mesh_view.set_polydata(poly_data)
        self.last_mesh_path = os.path.split(filename)[0]

    def print_info(self):
        print self.mesh_view.transform()
        for i in xrange(self.items.count()):
            print self._get_item(i)

    def add_correspondence(self):
        i, position = self.mesh_view.point()
        if i is None or position is None:
            return

        self._add_correspondence(position, i)
        self._set_correspondences()

    def _add_correspondence(self, position, i, is_active=True):
        self.items.addItem(self._make_item_string(position, i, is_active))

    def toggle_correspondence(self):
        row = self.items.currentRow()
        if row < 0:
            return

        is_active, (position, i) = self._get_item(row)
        self._update_correspondence(row, position, i, is_active ^ True)
        self._set_correspondences()

    def update_correspondence(self):
        row = self.items.currentRow()
        if row < 0:
            return

        is_active, (position, i) = self._get_item(row)
        _, position = self.mesh_view.point()
        self._update_correspondence(row, position, i, is_active)
        self._set_correspondences()

    def _update_correspondence(self, item_index, position, i, is_active):
        item = self.items.item(item_index)
        item.setText(self._make_item_string(position, i, is_active))

    def _set_correspondences(self):
        all_C = (self._get_item(i) for i in xrange(self.items.count()))
        active_C = filter(itemgetter(0), all_C)
        C = map(itemgetter(1), active_C)
        self.mesh_view.set_correspondences(C)
        
    def remove_correspondence(self):
        row = self.items.currentRow()
        if row < 0:
            return

        self.items.takeItem(row)
        self._set_correspondences()

    def get_projection_constraints(self):
        n = self.items.count()
        all_P = np.empty((n, 2), dtype=np.float64)
        all_C = np.empty((n,), dtype=np.int32)
        is_active = np.empty((n,), dtype=bool)

        for i in xrange(n):
            active, (position, point_id) = self._get_item(i)
            all_P[i] = position
            all_C[i] = point_id
            is_active[i] = active

        P = np.require(all_P[is_active], requirements='C')
        C = np.require(all_C[is_active], requirements='C')

        return dict(positions=P, point_ids=C,      # backwards compatible
                    C=C, P=P,
                    T=self.mesh_view.transform(),
                    V=self.mesh_view.V0,
                    all_P=all_P, all_C=all_C, is_active=is_active)
         
    def save(self):
        path = self.last_correspondences_path
        if path is None:
            path = QtCore.QDir.current().absolutePath()

        filename = QtGui.QFileDialog.getSaveFileName(self, 
            'Save Correspondences', path, '*.npz')

        if filename.isEmpty():
            return

        np.savez_compressed(str(filename), **self.get_projection_constraints())
        self.last_correspondences_path = os.path.split(str(filename))[0]

    def load(self):
        path = self.last_correspondences_path
        if path is None:
            path = QtCore.QDir.current().absolutePath()

        filename = QtGui.QFileDialog.getOpenFileName(self, 
            'Load Correspondences', path, '*.npz')

        if filename.isEmpty():
            return

        self._load(str(filename))

    def _load(self, filename):
        z = np.load(filename)

        # backwards compatibility
        if not 'all_P' in z.keys():
            positions = z['positions']
            point_ids = z['point_ids']
            is_active = np.ones(positions.shape[0], dtype=bool)
        else:
            positions = z['all_P']
            point_ids = z['all_C']
            is_active = z['is_active']
            
        for p, i, b in izip(positions, point_ids, is_active):
            self._add_correspondence(p, i, b)

        self._set_correspondences()
        self.mesh_view.set_transform(z['T'])

        self.last_correspondences_path = os.path.split(filename)[0]

    def toggle_arap_view(self):
        if self.toggle_arap_view_pb.isChecked():
            self.mesh_view.set_aux_polydata(
                self.arap_poly_data)
        else:
            self.mesh_view.set_aux_polydata(None)

    def apply_arap_deformation(self):
        # get projection constraints
        d = self.get_projection_constraints()

        # get maximum number of iterations
        maxIterations = int(str(self.max_iterations_le.text()))

        # prepare vertices
        V, T = d['V'], d['T']
        V = np.dot(V, np.transpose(T[:3,:3])) + T[:3,-1]

        # get topology (required for `solve_single_arap_proj`)
        T = self.mesh_view.triangles()

        # allocate rotations and initialise new vertices
        X = np.zeros_like(V)
        V1 = V.copy()

        lambdas = np.array([1.0,  # as-rigid-as-possible
                            1.0], # projection
                            dtype=np.float64)

        status, status_string = solve_single_arap_proj(
            V, T, X, V1, d['C'], d['P'], lambdas,
            maxIterations=maxIterations,
            gradientThreshold=1e-5,
            updateThreshold=1e-5,
            improvementThreshold=1e-5,
            verbosenessLevel=1)

        self.arap_poly_data = numpy_to_vtkPolyData(V1, faces_to_vtkCellArray(T))

    def _get_item(self, i):
        line = str(self.items.item(i).text())
        valid = line.split(':')[-1].lstrip()
        is_active, x, y, point_id = valid.split(',')

        is_active = (is_active == '1')
        x = float.fromhex(x)
        y = float.fromhex(y)
        point_id = int(point_id)

        return is_active, ((x, y), point_id)

    def _make_item_string(self, position, i, is_active):
        return ('[%s] %6.1f, %6.1f -> %4d : %s, %s, %s, %d' % 
                ('X' if is_active else ' ', position[0], position[1], i,
                 '1' if is_active else '0', position[0].hex(), position[1].hex(), i))

    def update_selection(self, row):
        if row < 0:
            return

        active, (position, point_id) = self._get_item(row)
        self.mesh_view._set_point(point_id)
        self.mesh_view._set_image_point(*position)

# main
def main():
    qapp = QtGui.QApplication([])

    main_window = MainWindow()
    main_window.show()

    # test
    # main_window._load_image('data/frames/dog0/dog-2741327_055.png')
    # main_window._load_mesh('data/models/Boxer_0B.npz')
    # main_window._load('data/user_constraints/dog0/Boxer_0B/55B.npz')

    qapp.exec_()

if __name__ == '__main__':
    main()


