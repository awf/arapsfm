# relabel_mesh.py

# Imports
import numpy as np
from vtk_ import *
from mesh.faces import *

# load_obj
def load_obj(filename):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(filename)
    reader.Update()

    poly_data = reader.GetOutput()

    V, T_ = vtkPolyData_to_numpy(poly_data)
    return V.copy(), list(iter_vtkCellArray(T_))

# remove_quads
def remove_quads(T):
    new_T = []
    for t in T:
        if len(t) == 4:
            new_T.append([t[0], t[1], t[2]])
            new_T.append([t[2], t[3], t[0]])
        else:
            new_T.append(list(t))

    return np.asarray(new_T, dtype=np.int32)

# relabel_mesh
def relabel_mesh(V, T):
    # build vertex -> [id] map
    vertex_to_ids = {}
    for t in T:
        for i in t:
            v = tuple(V[i])
            ids = vertex_to_ids.setdefault(v, set())
            ids.add(i)

    # build new vertex set and reverse mapping
    N = len(vertex_to_ids)
    V1 = np.empty((N, 3), dtype=np.float64)

    m = np.empty(V.shape[0], dtype=np.int32)
    for i, (v, s) in enumerate(vertex_to_ids.iteritems()):
        V1[i] = v
        for j in s:
            m[j] = i

    # build new triangles
    T1 = m[T]

    return V1, T1

# main_relabel_mesh
def main_relabel_mesh():
    V, T = load_obj('data/models/cat/Cartoon Cat/Cat.obj')
    print V.shape[0]
    T = remove_quads(T)

    V, T = relabel_mesh(V, T)
    print V.shape[0]
    T_ = faces_to_cell_array(T)

    poly_data = numpy_to_vtkPolyData(V, T_)
    view_vtkPolyData(poly_data)

    np.savez_compressed('data/models/cat/cat_fixed.npz', 
                        points=V, cells=T_,
                        V=V, T=T)

# main_smooth_model
def main_smooth_model():
    # load the fixed cat poly data
    z = np.load('data/models/cat/cat_fixed.npz')
    V = z['V']
    T = z['T']

    T_ = faces_to_cell_array(T)
    poly_data = numpy_to_vtkPolyData(V, T_)

    # get largest component
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetExtractionModeToLargestRegion()
    conn.SetInput(poly_data)

    # smooth
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(conn.GetOutputPort())
    smooth.SetRelaxationFactor(0.1)
    smooth.SetNumberOfIterations(30)

    # decimate
    deci = vtk.vtkDecimatePro()
    deci.SetTargetReduction(0.1)
    deci.SetInputConnection(smooth.GetOutputPort())
    deci.Update()

    poly_data = deci.GetOutput()

    V, T_ = vtkPolyData_to_numpy(poly_data)
    T = faces_from_cell_array(T_)
    print '# V:', V.shape[0]
    print '# T:', T.shape[0]

    view_vtkPolyData(poly_data)

    # save
    np.savez_compressed('data/models/cat/cat_simplified.npz',
                        points=V, cells=T_,
                        V=V, T=T)

if __name__ == '__main__':
    # main_relabel_mesh()
    main_smooth_model()

