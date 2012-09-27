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

// 
void lengthAdjustedSilhouetteProjResiduals_(PyArrayObject * npy_V1,
                                PyArrayObject * npy_q,
                                PyArrayObject * npy_S,
                                double w,
                                PyArrayObject * npy_e)
{
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_VECTOR(double, npy_q, q);
    PYARRAY_AS_VECTOR(double, npy_S, S);
    PYARRAY_AS_VECTOR(double, npy_e, e);

    lengthAdjustedSilhouetteProjResiduals_Unsafe(V1[0], V1[1], V1[2], 
                                                 q.begin(), &S[0], w, e.begin());
}

void lengthAdjustedSilhouetteProjJac_All_(PyArrayObject * npy_V1,
                                          PyArrayObject * npy_q,
                                          double w,
                                          PyArrayObject * npy_J)
{
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_VECTOR(double, npy_q, q);
    PYARRAY_AS_MATRIX(double, npy_J, J);

    double Ji[6], Jj[6], Jk[6], Jq[4]; 
    lengthAdjustedSilhouetteProjJac_V1i_Unsafe(V1[0], V1[2], q.begin(), w, Ji);
    lengthAdjustedSilhouetteProjJac_V1j_Unsafe(V1[1], V1[2], q.begin(), w, Jj);
    lengthAdjustedSilhouetteProjJac_V1k_Unsafe(V1[0], V1[1], V1[2], q.begin(), w, Jk);
    lengthAdjustedSilhouetteProjJac_q_Unsafe(V1[0], V1[1], V1[2], w, Jq);

    for (int i=0; i<2; i++)
    {
        for (int j=0; j<3; j++)
        {
            J[i][j] = Ji[3*i + j];
            J[i][3 + j] = Jj[3*i + j];
            J[i][6 + j] = Jk[3*i + j];
        }
        J[i][9] = Jq[2*i + 0];
        J[i][10] = Jq[2*i + 1];
    }
}

#endif
