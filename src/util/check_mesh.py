# check_mesh.py

# Imports
import argparse
import numpy as np
import mesh.faces
from vtk_ import iter_vtkCellArray, view_vtkPolyData, numpy_to_vtkPolyData
from operator import add
from itertools import repeat

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--check_edge_presence', 
                        default=False,
                        action='store_true')
    args = parser.parse_args()

    z = np.load(args.input)
    V = z['points']
    print '# Vertices:', V.shape[0]

    faces = list(iter_vtkCellArray(z['cells']))
    print '# Faces:', len(faces)

    n = np.unique(map(len, faces))
    print '# Vertices of on each face:', n

    edges = {}
    for f, face in enumerate(faces):
        for l in xrange(len(face)):
            i, j = face[l], face[(l + 1) % 3]
            edge = (i,j) if i < j else (j,i)
            edges.setdefault(edge, []).append(f)

    m = np.unique(map(len, edges.itervalues()))
    print '# Adjacent faces to each edge:', m

    highlight = set()
    for edge, face_indices in edges.iteritems():
        if len(face_indices) > 2:
            print face_indices
            for index in face_indices:
                map(highlight.add, faces[index])

    if args.check_edge_presence:
        # NOTE Redundant 
        half_edges = []
        for f, face in enumerate(faces):
            for l in xrange(len(face)):
                half_edges.append((face[l], face[(l + 1) % 3]))
                
        vertex_to_faces = mesh.faces.vertices_to_faces(V.shape[0], faces)

        all_edges = frozenset(edges.keys())

        for i in xrange(V.shape[0]):
            vertices_from_face = lambda j: filter(lambda k: k != i, faces[j])
            all_adj = map(vertices_from_face, vertex_to_faces[i])
            unique_adj = frozenset(reduce(add, all_adj))

            edges_should_be_present = map(lambda i,j: (i,j) if i < j else (j,i),
                                          repeat(i, len(unique_adj)), unique_adj)

            not_present = frozenset(edges_should_be_present).difference(all_edges)
            if not_present:
                print not_present
                map(highlight.add, not_present)

    min_, max_ = np.amin(V, axis=0), np.amax(V, axis=0)
    sphere_radius = 0.0125 * np.amin(max_ - min_)

    poly_data = numpy_to_vtkPolyData(V, z['cells'])
    view_vtkPolyData(poly_data, highlight=list(highlight), 
                     sphere_radius=sphere_radius)

if __name__ == '__main__':
    main()

