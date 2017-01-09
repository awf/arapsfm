# faces.py

# Imports
import numpy as np

# faces_from_cell_array
def faces_from_cell_array(cells):
    faces = []
    i = 0
    while i < cells.shape[0]:
        n = cells[i]
        faces.append(cells[i+1:i+n+1])
        i += n + 1

    return np.asarray(faces, dtype=np.int32)

# faces_to_cell_array
def faces_to_cell_array(faces):
    N = len(faces) + np.sum(map(len, faces))
    cells = np.empty(N, dtype=np.int32)

    i = 0
    for face in faces:
        n = len(face)
        cells[i] = n
        cells[i+1:i+n+1] = face

        i += n + 1

    return cells

# edges_in_face
def edges_in_face(face, preserve_direction):
    n = len(face)
    for i in xrange(n):
        edge = (face[i], face[(i+1) %n])

        if not preserve_direction:
            if edge[0] > edge[1]:
                edge = edge[::-1]

        yield edge

# adjacent_faces
def adjacent_faces(cells):
    faces = list(faces_from_cell_array(cells))
    N = len(faces)

    edge_to_faces = {}
    for i, face in enumerate(faces):
        for edge in edges_in_face(face):
            if edge[0] > edge[1]:
                edge = edge[::-1]

            face_indices = edge_to_faces.setdefault(edge, [])
            face_indices.append(i)

    adj = np.empty(N, dtype=np.ndarray)
    for i in xrange(N):
        adj[i] = set([])

    for f1, f2 in edge_to_faces.itervalues():
        adj[f1].add(f2)
        adj[f2].add(f1)

    for i in xrange(N):
        adj[i] = np.array(list(adj[i]))

    return adj

# vertices_to_faces
def vertices_to_faces(N, faces):
    mapping = np.empty(N, dtype=object)
    for i in xrange(N):
        mapping[i] = []

    for i, face in enumerate(faces):
        for l in face:
            mapping[l].append(i)

    return mapping

# vertices_to_triangles
def vertices_to_triangles(N, faces):
    mapping = vertices_to_faces(N, faces)

    lengths = np.array(map(len, mapping))
    if np.any(lengths == 0):
        raise ValueError('Some vertices do NOT match to any face!')

    vertices_to_triangles = np.hstack(mapping)
    offsets = np.r_[0, np.cumsum(map(len, mapping))]

    return vertices_to_triangles, offsets

# extended_one_rings
def extended_one_rings(N, faces):
    m = vertices_to_faces(N, faces)
    
    one_rings = []

    for face in faces:
        ext_ring = []

        for i in face:
            for j in m[i]:
                ext_ring += list(faces[j])

        one_rings.append(set(ext_ring))

    return one_rings
        
