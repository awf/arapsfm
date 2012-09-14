# user_constraints_ui.py 

# Imports
import numpy as np
import vtk

from matplotlib.pyplot import imread

from PyQt4 import QtGui, QtCore

from PyQt4_.imageview import ImageView
from PyQt4_.vtk.QVTKWidget import QVTKWidget

from itertools import izip
from vtk_ import *

from pprint import pprint

from os.path import splitext

# Constants
LAMBDAS = np.array([0, 1e4, 1e4, 1e-1, 1e2], dtype=np.float64)
MAXFUN = 1000

# GUI

# InteractiveMeshView
class InteractiveMeshView(QVTKWidget):
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
        
    def point(self):
        return self.last_point

    def transform(self):
        mat = self.box_transform.GetMatrix()
        if mat is None:
            T = np.eye(4)
        else:
            T = vtkMatrix4x4_to_numpy(mat)
        return T

    def reset(self):
        # reset poly data to original points
        if self.poly_data is not None:
            self.V[:] = self.V0[:]
            self.V0_prime[:] = self.V0[:]
            self.poly_data.Modified()

        # reset the transformation
        self.set_transform(np.eye(4))

    def set_transform(self, T):
        print 'Transform:'
        print T

        mat = numpy_to_vtkMatrix4x4(T)
        self.box_transform.SetMatrix(mat)
        self.box_widget.SetTransform(self.box_transform)

        self._transform_update()

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

        # get adjacency matrix and weights for the mesh
        self.cell_info = adj_vector(V, cells, 'cotan')

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

        # selection actors
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(5.0)

        sel_poly_data = vtk.vtkPolyData()
        sel_points = vtk.vtkPoints()
        sel_points.SetNumberOfPoints(1)
        sel_poly_data.SetPoints(sel_points)

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

    def _projection_update(self):
        proj_points = self.proj_poly_data.GetPoints()

        for j in xrange(len(self.C)):
            (x, y), i = self.C[j]
            proj_points.SetPoint(2*j + 1, self.V[i])

        self.proj_actor.SetVisibility(True)
        self.proj_poly_data.Modified()
        self.GetRenderWindow().Render()
            
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


