#ifndef __TEST_RESIDUALS_H__
#define __TEST_RESIDUALS_H__

#include "Geometry/mesh.h"
#include "Util/pyarray_conversion.h"
#include "Energy/narrow_band_silhouette.h"

// vertexNormal_
void vertexNormal_(PyArrayObject * npy_Triangles,
                   PyArrayObject * npy_V1,
                   int vertexId,
                   double * normal)
{
    PYARRAY_AS_MATRIX(int, npy_Triangles, Triangles);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);

    Mesh mesh(V1.num_rows(), Triangles);

    vertexNormal_Unsafe(mesh, V1, vertexId, normal);
}

// vertexNormalJac_
PyArrayObject * vertexNormalJac_(PyArrayObject * npy_Triangles,
                   PyArrayObject * npy_V1,
                   int vertexId)
{
    PYARRAY_AS_MATRIX(int, npy_Triangles, Triangles);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);

    Mesh mesh(V1.num_rows(), Triangles);

    Matrix<double> J = vertexNormalJac(mesh, V1, vertexId);

    npy_intp dims[2] = {J.num_rows(), J.num_cols()};

    PyArrayObject * npy_J = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    std::copy(J.begin(), J.end(), (double *)PyArray_DATA(npy_J));

    return npy_J;
}

void silhouetteNormalResiduals_(PyArrayObject * npy_Triangles,
                                PyArrayObject * npy_V1,
                                int faceIndex,
                                PyArrayObject * npy_u,
                                PyArrayObject * npy_SN,
                                double w,
                                PyArrayObject * npy_e)
{
    PYARRAY_AS_MATRIX(int, npy_Triangles, Triangles);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_VECTOR(double, npy_SN, SN);
    PYARRAY_AS_VECTOR(double, npy_u, u);
    PYARRAY_AS_VECTOR(double, npy_e, e);

    Mesh mesh(V1.num_rows(), Triangles);

    silhouetteNormalResiduals_Unsafe(mesh, V1, faceIndex, u.begin(),
        SN.begin(), w, e.begin());
}

// silhouetteNormalResidualsJac_V1_
PyArrayObject * silhouetteNormalResidualsJac_V1_(PyArrayObject * npy_Triangles,
                                                 PyArrayObject * npy_V1,
                                                 int faceIndex,
                                                 PyArrayObject * npy_u,
                                                 int vertexIndex,
                                                 double w)
{
    PYARRAY_AS_MATRIX(int, npy_Triangles, Triangles);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_VECTOR(double, npy_u, u);

    Mesh mesh(V1.num_rows(), Triangles);

    npy_intp dims[2] = {3, 3};
    PyArrayObject * npy_J = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    silhouetteNormalResidualsJac_V1_Unsafe(mesh, V1, faceIndex, u.begin(),
        vertexIndex, w, (double *)PyArray_DATA(npy_J));

    return npy_J;
}

void silhouetteNormalResidualsJac_u_(PyArrayObject * npy_Triangles,
                                     PyArrayObject * npy_V1,
                                     int faceIndex,
                                     PyArrayObject * npy_u,
                                     double w,
                                     double * J)
{

    PYARRAY_AS_MATRIX(int, npy_Triangles, Triangles);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_VECTOR(double, npy_u, u);

    Mesh mesh(V1.num_rows(), Triangles);

    silhouetteNormalResidualsJac_u_Unsafe(mesh, V1, faceIndex, u.begin(), w, J);
}

#endif
