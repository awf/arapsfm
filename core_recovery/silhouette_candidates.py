# silhouette_candidates.py

# Imports
from __future__ import division
import numpy as np

from scipy.linalg import norm
import heapq
from itertools import count, chain
from operator import itemgetter

from mesh.faces import edges_in_face

# Visualisation
from visualise.vtk_ import *

# view_with_distance_matrix
def view_with_distance_matrix(V, cells, Q, D, N, camera_opt={}, cone_opt={}):
    V = np.asarray(V, dtype=np.float32)
    model_pd = numpy_to_vtkPolyData(V, cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(model_pd)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.26, 0.58, 0.76)

    Q = np.asarray(Q, dtype=np.float32)
    N = np.asarray(N, dtype=np.float32)

    vtk_Q = numpy_to_vtk(Q)
    vtk_N = numpy_to_vtk(N)

    geodesic_points = vtk.vtkPoints()
    geodesic_points.SetData(vtk_Q)
    geodesics_pd = vtk.vtkPolyData()
    geodesics_pd.GetPointData().SetNormals(vtk_N)
    geodesics_pd.SetPoints(geodesic_points)

    cone = vtk.vtkConeSource()

    for option, args in cone_opt.iteritems():
        method_name = 'Set%s' % option
        try:
            method = getattr(cone, method_name)
        except :
            pass
        method(args)

    cone.SetCenter(cone.GetHeight() / 2.0,0,0)

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(cone.GetOutputPort())
    glyph.SetInput(geodesics_pd)

    glyph.SetVectorModeToUseNormal()
    glyph.SetScaleModeToDataScalingOff()

    glyph_mapper = vtk.vtkPolyDataMapper()
    glyph_mapper.SetInputConnection(glyph.GetOutputPort())

    glyph_actor = vtk.vtkActor()
    glyph_actor.SetMapper(glyph_mapper)
    glyph_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    glyph_actor.PickableOff()

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.AddActor(glyph_actor)
    ren.SetBackground(1.0, 1.0, 1.0)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(ren)
    render_window.SetSize(800, 800)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    camera = ren.GetActiveCamera()
    _camera_opt = {'FocalPoint':(0,0,0), 'Position':(0, 0, 1),
                   'ViewUp':(0,1,0)}
    _camera_opt.update(camera_opt)

    for property_, args in _camera_opt.iteritems():
        method_name = 'Set%s' % property_
        try:
            method = getattr(camera, method_name)
        except :
            pass

        method(args)

    ren.ResetCamera()

    picker = vtk.vtkPointPicker()

    def annotatePick(obj, event):
        i = picker.GetPointId()
        if i == -1:
            return

        print 'i:', i
        print 'argmin(D[i]):', np.argsort(D[i])[:5], ' ...'
        d = D[i].copy()
        d /= np.amax(d)
        d = 1.0 - d

        vtk_d = numpy_to_vtk(d)
        vtk_d._npy = d

        geodesics_pd.GetPointData().SetScalars(vtk_d)
        geodesics_pd.Modified()

        glyph.SetColorModeToColorByScalar()

        render_window.Render()

    picker.AddObserver('EndPickEvent', annotatePick)

    iren.SetPicker(picker)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetCurrentRenderer(ren)
    iren.SetInteractorStyle(style)

    iren.Initialize()
    iren.Start()

# Triangular grid (unused)

# subdivide_triangle
def subdivide_triangle(num_candidates):
    candidates = []

    to_process = [(np.eye(3, dtype=np.float64), num_candidates)]

    while to_process:
        t, n = to_process.pop(0)

        if n <= 1:
            candidates.append(np.mean(t, axis=0))
            continue

        m = np.empty((3, 3), dtype=np.float64)
        for i in xrange(3):
            m[i] = 0.5 * (t[i] + t[(i+1) % 3])

        n = n // 4
        to_process.append((np.array([t[0], m[0], m[2]]), n))
        to_process.append((np.array([t[1], m[1], m[0]]), n))
        to_process.append((np.array([t[2], m[2], m[1]]), n))
        to_process.append((np.array([m[0], m[1], m[2]]), n))

    return np.array(candidates)

# triangular_grid
def triangular_grid(N, d=None):
    x = np.roots(np.array([1.0, 1.0, -2.0*N]))
    n = np.ceil(np.max(x)).astype(int)
    if n < 2:
        return np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    if d is None:
        d = (1.0/(n+1)) / (0.5 * np.sqrt(3))

    v = d*0.5*np.sqrt(3)
    step = (1 - 2*v) / (n-1)
    u = v
    v = v - d/(2*np.sqrt(3))
    
    X = np.empty(((n*n+n)/2, 2), dtype=np.float64)

    j = 0
    for i in xrange(n):
        m = n - i
        if m > 1:
            u_step = (1.0 - 2*u) / (m - 1)
            U = u + np.arange(0, m)*u_step
        else:
            U = 0.5

        X[j:(j+m), 1] = v
        X[j:(j+m), 0] = U - 0.5*v

        j += m
        v += step
        u += 0.5*step

    return X

# Djikstra with only edge points and vertices

# edist_matrix
def edist_matrix(V):
    N = V.shape[0]
    D = np.zeros((N, N), dtype=np.float64)

    for i, vi in enumerate(V):
        for j, vj in enumerate(V):
            if i < j: continue

            D[i,j] = norm(vi - vj)

    return D + np.transpose(D)

# generate_silhouette_candidate_info
def generate_silhouette_candidate_info(V, faces, step=np.inf, verbose=True):
    # form the new candidate edge points
    edge_points = []                # list of arrays of new edge points
    edge_points_info = []           # list of (edge, t)
    vertex_to_faces = {}            # vertex -> faces it contributes 
    edge_candidates = {}            # edge -> list of indices into `Q`
    edge_from_face = {}             # edge -> (face, barycentric) it is from

    edge_points_face_info = []      # list of (face, u [barycentric])
    vertex_face_info = []           # list of (face, u [barycentric])

    edge_to_faces = {}              # edge -> list of faces which it contributes to

    face_normals = np.empty((len(faces), 3), dtype=np.float64)

    index = V.shape[0]              # starting index for the new edge points

    for i, face in enumerate(faces):
        # calculate face normal
        Vi = V[face]
        Vi02 = Vi[0] - Vi[2]
        Vi12 = Vi[1] - Vi[2]
        n = np.cross(Vi02, Vi12)
        face_normals[i, :] = n / norm(n)

        # update vertex -> faces
        for v in face:
            vertex_to_faces.setdefault(v, []).append(i)

        for j, edge in enumerate(edges_in_face(face, True)):
            edge_no_direction = edge if edge[0] < edge[1] else edge[::-1]
            edge_to_faces.setdefault(edge_no_direction, []).append((i, j))

            # divide edge
            if edge not in edge_candidates and edge[::-1] not in edge_candidates:
                V0, V1 = V[list(edge)]
                length = norm(V0 - V1)
                n = np.floor(length / step).astype(int)

                if n >= 1:
                    t = np.linspace(0., 1., n+2, endpoint=True)[1:-1]
                    e = np.outer(1-t, V0) + np.outer(t, V1)
                    edge_points.append(e)

                    for t_ in t:
                        edge_points_info.append((edge, t_))

                    edge_candidates[edge] = range(index, index+n)

                    # determine barycentric coordinates in the source face
                    U_bary = np.empty((n, 3), dtype=np.float64)
                    U_bary[:, j] = 1 - t
                    U_bary[:, (j + 1) % 3] = t
                    U_bary[:, (j + 2) % 3] = (1.0 - U_bary[:, (j + 1) % 3] - 
                                              U_bary[:, j])

                    for l in xrange(n):
                        edge_points_face_info.append((i, U_bary[l, :2]))
                    
                    index += n
                else:
                    edge_candidates[edge] = []

    # form array of augmented vertices
    Q = np.vstack([V] + edge_points)

    # form array of normals
    N = np.zeros_like(Q)

    # normals at vertices
    for i in xrange(V.shape[0]):
        face_indices = vertex_to_faces[i]
        adjacent_face_normals = face_normals[face_indices]
        normal = np.mean(adjacent_face_normals, axis=0)
        normal /= norm(normal)
        N[i,:] = normal

        # assign vertices to the faces which have closest normal to the mean
        diff = normal[np.newaxis, :] - adjacent_face_normals
        sqr_diff = np.sum(diff*diff, axis=1)
        best_face = face_indices[np.argmin(sqr_diff)]

        # determine the barycentric coordinates for this face
        u = np.zeros(3, dtype=np.float64)
        u[faces[best_face] == i] = 1.0

        # save information
        vertex_face_info.append((best_face, u[:2]))

    # normals at new edge points are interpolated from the vertices
    for i, (edge, t) in enumerate(edge_points_info, start=V.shape[0]):
        v_norms = N[list(edge)]
        normal = (1-t) * v_norms[0] + t * v_norms[1]
        N[i,:] = normal / norm(normal)

    # build information required for reconstruction of vertices and normals
    info = {}
    info['SilEdgeCands'] = np.array([edge for edge, t in edge_points_info], dtype=np.int32)
    info['SilEdgeCandParam'] = np.array([t for edge, t in edge_points_info], dtype=np.float64)

    all_face_info = vertex_face_info + edge_points_face_info
    info['SilCandAssignedFaces'] = np.array(map(itemgetter(0), all_face_info),
                                            dtype=np.int32)
    info['SilCandU'] = np.vstack(map(itemgetter(1), all_face_info))

    # build adjacency information
    adj = np.empty(Q.shape[0], dtype=object)
    for i in xrange(adj.shape[0]):
        adj[i] = set([])
    W = np.zeros((Q.shape[0], Q.shape[0]), dtype=np.float64)

    for i, face in enumerate(faces):
        # get all candidates within a face
        I = set(face)
        for edge in edges_in_face(face, preserve_direction=True):
            try:
                candidates = edge_candidates[edge]
            except KeyError:
                candidates = edge_candidates[edge[::-1]]

            map(I.add, candidates)

        # calculate within face euclidean distance matrix and store
        I = list(I)
        W[np.ix_(I, I)] = edist_matrix(Q[I])

        # add adjacency information
        n = len(I)
        for l, j in enumerate(I):
            map(adj[j].add, I[:l] + I[l+1:])

    # apply sequential Djikstra
    D = np.empty((Q.shape[0], Q.shape[0]), dtype=np.float64)
    D.fill(np.inf)
    np.fill_diagonal(D, 0)

    if verbose:
        print 'Applying sequential Djikstra (%d)' % (Q.shape[0])

    for s in xrange(Q.shape[0]):
        if verbose:
            print ' %d' % s

        # initialise min-heap
        to_process = []
        counter = count(0)
        processed = np.zeros(Q.shape[0], dtype=np.uint8)

        to_process.append((0., 0, s))

        for l in xrange(Q.shape[0]):
            while to_process:
                d, n, i = heapq.heappop(to_process)
                if not processed[i]:
                    break
            else:
                raise ValueError('to_process is empty!')

            for j in adj[i]:
                D[s,j] = min(D[s,i] + W[i,j], D[s,j])
                if not processed[j]:
                    heapq.heappush(to_process, (D[s,j], next(counter), j))

            processed[i] = 1

    # update info with the distance matrix
    info['SilCandDistances'] = D

    return info

