# user_constraints_ui.py 

# Imports
import numpy as np

import vtk
from visualise.vtk_ import *

from PyQt4 import QtGui, QtCore
from mesh_view import InteractiveMeshView

from matplotlib.pyplot import imread
from itertools import izip
from pprint import pprint
from os.path import splitext

# GUI

# MainWindow
class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.setup_ui()

    def setup_ui(self):
        self.load_image_pb = QtGui.QPushButton('Load &Image')
        self.load_mesh_pb = QtGui.QPushButton('Load &Mesh')
        self.load_pb = QtGui.QPushButton('&Load')
        self.save_pb = QtGui.QPushButton('&Save')
        self.add_pb = QtGui.QPushButton('&Add')
        self.remove_pb = QtGui.QPushButton('&Remove')
        self.reset_pb = QtGui.QPushButton('Rese&t')
        self.print_info_pb = QtGui.QPushButton('&Print Info')

        pb_layout = QtGui.QGridLayout()
        pb_layout.addWidget(self.load_image_pb, 0, 0)
        pb_layout.addWidget(self.load_mesh_pb, 0, 1)
        pb_layout.addWidget(self.add_pb, 1, 0)
        pb_layout.addWidget(self.remove_pb, 1, 1)
        pb_layout.addWidget(self.save_pb, 2, 0)
        pb_layout.addWidget(self.load_pb, 2, 1)
        pb_layout.addWidget(self.reset_pb, 3, 0)
        pb_layout.addWidget(self.print_info_pb, 3, 1)

        self.items = QtGui.QListWidget()
        ctrl_layout = QtGui.QVBoxLayout()
        ctrl_layout.addWidget(self.items)
        ctrl_layout.addLayout(pb_layout)

        self.mesh_view = InteractiveMeshView()

        view_layout = QtGui.QVBoxLayout()
        view_layout.addWidget(self.mesh_view)

        main_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(ctrl_layout)
        main_layout.addLayout(view_layout, stretch=1)

        main_widget = QtGui.QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.load_image_pb.clicked.connect(self.load_image)
        self.load_mesh_pb.clicked.connect(self.load_mesh)
        self.add_pb.clicked.connect(self.add_correspondence)
        self.remove_pb.clicked.connect(self.remove_correspondence)
        self.save_pb.clicked.connect(self.save)
        self.load_pb.clicked.connect(self.load)
        self.print_info_pb.clicked.connect(self.print_info)

        self.reset_pb.clicked.connect(self.mesh_view.reset)

        self.items.currentRowChanged.connect(self.update_selection)

        self.setWindowTitle('DeformationPlay')

    def load_image(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Load Image',
            QtCore.QDir.current().absolutePath(),
            '*.png')

        if filename.isEmpty():
            return

        self.mesh_view.set_image(str(filename))

    def load_mesh(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Load Mesh',
            QtCore.QDir.current().absolutePath(),
            'Meshes (*.stl *.npz)')

        if filename.isEmpty():
            return

        filename = str(filename)
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

    def _add_correspondence(self, position, i):
        item_string = ('(%.3f, %.3f) -> %d : %s, %s, %d' % 
                      (position[0], position[1], i,
                       position[0].hex(), position[1].hex(),
                       i))

        self.items.addItem(item_string)

    def _set_correspondences(self):
        C = [self._get_item(i) for i in xrange(self.items.count())]
        self.mesh_view.set_correspondences(C)
        
    def remove_correspondence(self):
        row = self.items.currentRow()
        if row < 0:
            return

        self.items.takeItem(row)
        self._set_correspondences()

    def save(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, 
            'Save Correspondences',
            QtCore.QDir.current().absolutePath(),
            '*.npz')

        if filename.isEmpty():
            return

        positions, point_ids = [], []

        for i in xrange(self.items.count()):
            position, point_id = self._get_item(i)
            positions.append(position)
            point_ids.append(point_id)

        np.savez_compressed(str(filename),
            positions=np.asarray(positions, dtype=float),
            point_ids=np.asarray(point_ids, dtype=int),
            T=self.mesh_view.transform(),
            V=self.mesh_view.V0)

    def load(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 
            'Load Correspondences',
            QtCore.QDir.current().absolutePath(),
            '*.npz')

        if filename.isEmpty():
            return

        z = np.load(str(filename))
        positions = z['positions']
        point_ids = z['point_ids']
        T = z['T']

        for position, point_id in izip(positions, point_ids):
            self._add_correspondence(position, point_id)

        self._set_correspondences()
        self.mesh_view.set_transform(T)

    def _get_item(self, i):
        line = str(self.items.item(i).text())
        valid = line.split(':')[-1].lstrip()
        row, col, point_id = valid.split(',')

        row = float.fromhex(row)
        col = float.fromhex(col)
        point_id = int(point_id)

        return (row, col), point_id

    def update_selection(self, row):
        if row < 0:
            return

        position, point_id = self._get_item(row)
        self.mesh_view._set_point(point_id)
        self.mesh_view._set_image_point(*position)

# main
def main():
    qapp = QtGui.QApplication([])

    main_window = MainWindow()
    main_window.show()

    qapp.exec_()

if __name__ == '__main__':
    main()


