# loop.py

# Imports
import numpy as np
import itertools as it
from scipy import sparse
import time

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import vtk
from vtk.util.numpy_support import vtk_to_numpy

# Iterators

# directed_edges
def directed_edges(t):
    for l in xrange(3):
        yield t[l], t[(l + 1) % 3]

# edges
def edges(t):
    for i, j in directed_edges(t):
        yield (i, j) if i < j else (j, i)

# csr_rows
def csr_rows(A):
    for i in xrange(A.shape[0]):
        yield A.indices[A.indptr[i]:A.indptr[i+1]]
            
# Subdivision

# subdivide
def subdivide(T, V):
    N = V.shape[0]
    adj_matrix = sparse.lil_matrix((N, N), dtype=np.uint8)
    edge_vertex_info = {}
    edges_processed = []
    child_T = np.empty_like(T, dtype=np.int32)

    for t_index, t in enumerate(T):
        for e_index, edge in enumerate(edges(t)):
            # Define new vertex on the edge
            if edge not in edge_vertex_info:
                edge_vertex_info[edge] = (len(edges_processed) + N, [])
                edges_processed.append(edge)

            # Save the edge vertex index
            l, opp_vertices = edge_vertex_info[edge]
            child_T[t_index, e_index] = l

            # Save the vertex oppositive the edge
            opp_vertices.append(t[e_index - 1])

            # Save adjacency neighbourhood information
            i, j = edge
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    adj_matrix = adj_matrix.tocsr()

    V1 = np.empty((N + len(edges_processed), 3), dtype=V.dtype)

    # Generate new positions of current vertices
    for i, j in enumerate(csr_rows(adj_matrix)):
        n = len(j)
        if n < 2:
            raise ValueError('n < 2')
        elif n == 2:
            w = 0.125
            w0 = 0.75
        elif n == 3:
            w = 3.0 / 16.0
            w0 = 7.0 / 16.0
        else:
            w = 3.0 / (8.0 * n)
            w0 = 1.0 - n * w

        V1[i] = w0 * V[i] + w * np.sum(V[j], axis=0)

    # Generate positions of the new vertices generated at edges
    for l, edge in enumerate(edges_processed, start=N):
        op = edge_vertex_info[edge][1]
        i, j = edge

        n = len(op)
        if n == 1:
            V1[l] = 0.5 * (V[i] + V[j])
        elif n == 2:
            V1[l] = 0.375 * (V[i] + V[j]) + 0.125 * (V[op[0]] + V[op[1]])
        else:
            raise ValueError('n > 2')

    # Generate new triangles
    T1 = np.empty((4*T.shape[0], 3), dtype=T.dtype)
    for i, t in enumerate(T):
        c = child_T[i]

        T1[4*i,   :] = (t[0], c[0], c[2])
        T1[4*i+1, :] = (t[1], c[1], c[0])
        T1[4*i+2, :] = (t[2], c[2], c[1])
        T1[4*i+3, :] = (c[0], c[1], c[2])

    return T1, V1

# Visualisation

# plot_mesh_2d
def plot_mesh_2d(ax, T, V, *args, **kwargs):
    N = V.shape[0]
    edge_plotted = sparse.lil_matrix((N, N), dtype=np.uint8)

    for t in T:
        for i, j in edges(t):
            if not edge_plotted[i, j]:  
                ax.plot([V[i, 0], V[j, 0]], [V[i, 1], V[j, 1]], 
                        *args, **kwargs)
                edge_plotted[i, j] = 1

# main_triangle_2D
def main_triangle_2D():
    # Setup `T` (triangles) and `V` (vertices)
    T = np.r_[0, 1, 2].astype(np.int32).reshape(-1, 3)
    V = np.r_[-0.5, 0, 0,
              0., np.sqrt(3) / 2.0, 0,
              0.5, 0, 0].astype(np.float64).reshape(-1, 3)

    # Plot original triangles
    f = plt.figure()
    ax = f.add_subplot(111, aspect='equal')
    plot_mesh_2d(ax, T, V, 'o-', c=(1.0, 0., 1.0, 1.0))

    # Subdivide `n` times
    n = 4

    t1 = time.time()
    subdivided = []
    for i in xrange(n):
        T, V = subdivide(T, V)
        subdivided.append((T, V))
    t2 = time.time()
    print 'Time taken: %.3fs' % (t2 - t1)

    colors = cm.jet(np.linspace(0., 1., n, endpoint=True))
    for i, (T, V) in enumerate(subdivided):
        plot_mesh_2d(ax, T, V, 'o-', c=colors[i])
        
    min_, max_ = np.amin(V, axis=0), np.amax(V, axis=0)
    ax.set_xlim(min_[0] - 0.5, max_[0] + 0.5)
    ax.set_ylim(min_[1] - 0.5, max_[1] + 0.5)
    plt.show()

# triangles_to_vtkCellArray
def triangles_to_vtkCellArray(T):
    N = len(faces) + np.sum(map(len, T))
    T_ = np.empty(N, dtype=np.int32)

    i = 0
    for t in T:
        n = len(t)
        T_[i] = n
        T_[i+1:i+n+1] = t
        i += n + 1

    return T_

# numpy_to_vtkPolyData
def numpy_to_vtkPolyData(T, V):
    vtk_cells = vtk.vtkCellArray()
    for t in T:
        vtk_cells.InsertNextCell(t.shape[0])
        for i in t:
            vtk_cells.InsertCellPoint(i)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(V.shape[0])
    npy_points = vtk_to_numpy(vtk_points.GetData())
    npy_points.flat = V.flat

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetPolys(vtk_cells)

    return poly_data

# octohedron
def octohedron():
    """From http://prideout.net/blog/?p=44"""
    f = np.sqrt(2.0) / 2.0
    V = np.array([( 0, -1,  0),
                  (-f,  0,  f),
                  ( f,  0,  f),
                  ( f,  0, -f),
                  (-f,  0, -f),
                  ( 0,  1,  0)], dtype=np.float64)
    T = np.array([(0, 2, 1),
                  (0, 3, 2),
                  (0, 4, 3),
                  (0, 1, 4),
                  (5, 1, 2),
                  (5, 2, 3),
                  (5, 3, 4),
                  (5, 4, 1)], dtype=np.int32)

    return T, V
    
# main_octohedron
def main_octohedron():
    # Setup renderer and `add_mesh` function
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)

    def add_mesh(T, V, **kwargs):
        pd = numpy_to_vtkPolyData(T, V)

        pd_normals = vtk.vtkPolyDataNormals()
        pd_normals.SetInput(pd)
        pd_normals.ComputeCellNormalsOn()
        pd_normals.SetFeatureAngle(90.)

        pd_depth = vtk.vtkDepthSortPolyData()
        pd_depth.SetInputConnection(pd_normals.GetOutputPort())
        pd_depth.SetCamera(ren.GetActiveCamera())
        pd_depth.SetDirectionToBackToFront()
        pd_depth._input = pd_normals

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(pd_depth.GetOutputPort())
        mapper._input = pd_depth

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor_property = actor.GetProperty()

        for prop, args in kwargs.iteritems():
            method = getattr(actor_property, 'Set' + prop)
            method(*args)

        ren.AddActor(actor)

    # Set number of subdivision levels `n` and add to the visualisation
    n = 3
    colors = cm.Accent(np.linspace(0., 1., n + 1, endpoint=True))[:, :3]

    T, V = octohedron()
    add_mesh(T, V, Color=colors[0], Opacity=(0.1,), EdgeVisibility=(True,))

    for i in xrange(1, n + 1):
        T, V = subdivide(T, V)
        add_mesh(T, V, 
                 Color=colors[i], 
                 Opacity=(0.1 if i < n else 1.0,),
                 EdgeVisibility=(i < n, )) 

    # Setup the render window and interactor
    ren.ResetCamera()

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(ren)
    render_window.SetSize(500, 500)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetCurrentRenderer(ren)
    iren.SetInteractorStyle(style)

    iren.Initialize()
    iren.Start()

if __name__ == '__main__':
    # main_triangle_2D()
    main_octohedron()

