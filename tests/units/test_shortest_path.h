#ifndef __TEST_SHORTEST_PATH_H__
#define __TEST_SHORTEST_PATH_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Silhouette/shortest_path.h"
#include "Util/pyarray_conversion.h"

// test_shortest_path
PyArrayObject * test_shortest_path(PyArrayObject * npy_V,
                       PyArrayObject * npy_T,
                       PyArrayObject * npy_S,
                       PyArrayObject * npy_SN,
                       PyArrayObject * npy_SilCandDistances,
                       PyArrayObject * npy_SilEdgeCands,
                       PyArrayObject * npy_SilEdgeCandParam,
                       PyArrayObject * npy_SilCandAssignedFaces,
                       PyArrayObject * npy_SilCandU,
                       PyArrayObject * npy_lambdas,
                       bool isCircular)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_S, S);
    PYARRAY_AS_MATRIX(double, npy_SN, SN);

    PYARRAY_AS_MATRIX(double, npy_SilCandDistances, SilCandDistances);
    PYARRAY_AS_MATRIX(int, npy_SilEdgeCands, SilEdgeCands);
    PYARRAY_AS_VECTOR(double, npy_SilEdgeCandParam, SilEdgeCandParam);
    PYARRAY_AS_VECTOR(int, npy_SilCandAssignedFaces, SilCandAssignedFaces);
    PYARRAY_AS_MATRIX(double, npy_SilCandU, SilCandU);
    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    Mesh mesh(V.num_rows(), T);
    ShortestPathInfo info(mesh, SilCandDistances, SilEdgeCands, SilEdgeCandParam, SilCandAssignedFaces, SilCandU);

    ShortestPathSolver solver(S, SN, info, lambdas, true);

    npy_intp dim = solver.getPathLength();
    PyArrayObject * npy_path = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_INT32);
    PYARRAY_AS_VECTOR(int, npy_path, path);

    solver.updateCandidates(V);
    solver.solve(path, isCircular);

    return npy_path;
}

#endif

