# box_model.py

# Imports
import numpy as np
from visualise import *
from mesh.faces import faces_to_cell_array
from vtk_ import *

# Box construction

# quads_to_triangles
def quads_to_triangles(quads):
    triangles = []
    for quad in quads:
        triangles.append(quad[:3])
        triangles.append(quad[2:] + quad[:1])

    return triangles
    
# box_end_layer
def box_end_layer(N, indices):
    n = N + 1

    quads = []
    for i in xrange(N):
        for j in xrange(N):
            quad = [indices[i*n + j],
                    indices[i*n + j + 1],
                    indices[(i+1)*n + j + 1],
                    indices[(i+1)*n + j]]
            quads.append(quad)

    return quads_to_triangles(quads)

# box_end_layer_vertices
def box_end_layer_vertices(N, w, h):
    n = N + 1

    v = np.empty((n*n, 3), dtype=np.float64)
    for i in xrange(n):
        for j in xrange(n):
            v[i*n + j] = (w*j, w*(n-i-1), h)

    return v

# box_intermediate_layer
def box_intermediate_layer(N, last_indices, indices):
    quads  = []
    l = len(indices)
    for i in xrange(4*N):
        quad = [last_indices[i],
                indices[i],
                indices[(i+1) % l],
                last_indices[(i+1) % l]]
        quads.append(quad)

    return quads_to_triangles(quads)

# box_intermediate_layer_vertices
def box_intermediate_layer_vertices(N, w, h):
    n = N + 1

    v = np.empty((4*N, 3), dtype=np.float64)
    for i in xrange(n):
        v[i] = (w*i, w*(n-1), h)
    for i in xrange(N-1):
        v[i+n] = (w*(n-1), w*((n-1)-(i+1)), h)
    for i in xrange(n):
        v[i+n+N-1] = (w*((n-1) - i), 0, h)
    for i in xrange(N-1):
        v[i+2*n+N-1] = (0, w*(i+1), h)

    return v

# end_layer_border
def end_layer_border(N, layer_indices):
    n = N + 1

    border = []
    border += layer_indices[:n]
    for i in xrange(1, N):
        border.append(layer_indices[i*n + (n-1)])
    border += layer_indices[N*n:][::-1]
    for i in xrange(N-1, 0, -1):
        border.append(layer_indices[i*n])

    return border

# box_model
def box_model(N, H, w, h):
    n = N + 1           # number of vertices on a side
    num_end_layer = n*n # number of vertices on the bottom/top layers   
    num_int_layer = 4*N # number of vertices in an intermediate layer
    num_vertices = 2*num_end_layer + (H-2)*num_int_layer    # H is number of
                                                            # layers
    def height(i):
        try:
            return h[i]
        except TypeError:
            return (i+1)*h

    faces = []
    V = np.empty((num_vertices, 3), dtype=np.float64)
    V.fill(-1.)

    # bottom layer
    bottom_indices = range(num_end_layer)

    bottom = box_end_layer(N, bottom_indices)

    faces += bottom
    V[bottom_indices] = box_end_layer_vertices(N, w, height(0))

    # intermediate layers
    indices = end_layer_border(N, bottom_indices)
    l = num_end_layer

    for i in xrange(H-2):
        last_indices = indices
        indices = range(l, l+4*N)

        layer = box_intermediate_layer(N, last_indices, indices)
        faces += layer

        V[indices] = box_intermediate_layer_vertices(N, w,
            height(i+1))

        l += 4*N

    # top layer
    top_indices = range(l, l+num_end_layer)
    top_border = end_layer_border(N, top_indices)
    layer = box_intermediate_layer(N, indices, top_border)
    faces += layer

    top = box_end_layer(N, top_indices)
    faces += top
    V[top_indices] = box_end_layer_vertices(N, w, height(H-1))

    return V, np.asarray(faces, dtype=np.int32)

# main
def main():
    if not HAS_VTK:
        return

    V, T = box_model(5,10, 1.0, 1.0)
    poly_data = numpy_to_vtkPolyData(V, faces_to_cell_array(T))
    view_vtkPolyData(poly_data)

    # vis = VisualiseMesh()
    # vis.add_mesh(V, T)
    # vis.execute()

if __name__ == '__main__':
    main()

