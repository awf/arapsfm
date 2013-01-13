#ifndef __TEST_SILHOUETTE_H__
#define __TEST_SILHOUETTE_H__

#include "Geometry/mesh.h"
#include "Energy/narrow_band_silhouette.h"
#include "Util/pyarray_conversion.h"

#include <iostream>
using namespace std;

// from `cpp/Energy/narrow_band_silhouette.h`

// normalizationJac_Unsafe
inline void normalizationJac_Unsafe(const double * v, double * J)
{
    double x[10];

    x[0] = v[0]*v[0];
    x[1] = v[1]*v[1];
    x[2] = v[2]*v[2];
    x[3] = x[0] + x[1] + x[2];
    x[5] = 1.0 / sqrt(x[3]);
    x[4] = x[5]*x[5]*x[5];
    //x[4] = x[3]**(-3/2);
    //x[5] = x[4]**(1/3);
    x[6] = v[2]*x[4];
    x[7] = v[0]*v[1]*x[4];
    x[8] = v[0]*x[6];
    x[9] = v[1]*x[6];

    J[0] = -x[0]*x[4] + x[5];
    J[1] = -x[7];
    J[2] = -x[8];
    J[3] = -x[7];
    J[4] = -x[1]*x[4] + x[5];
    J[5] = -x[9];
    J[6] = -x[8];
    J[7] = -x[9];
    J[8] = -x[2]*x[4] + x[5];
}

inline void lengthJac_Unsafe(const double * v, double * J)
{
    double x[5];
    x[0] = v[0]*v[0];
    x[1] = v[1]*v[1];
    x[2] = v[2]*v[2];
    x[3] = x[0] + x[1] + x[2];
    x[4] = 1.0 / sqrt(x[3]);

    J[0] = v[0] * x[4];
    J[1] = v[1] * x[4];
    J[2] = v[2] * x[4];
}

void silhouetteNormalResiduals2_Unsafe(const Mesh & mesh, const Matrix<double> & V1, 
                                       int faceIndex, const double * u_, const double * SN, 
                                       const double & w, double * e)
{
    const int * Ti = mesh.GetTriangle(faceIndex);

    double vertexNormals[3][3];
    double vertexNormalLengths[3];

    for (int i=0; i < 3; i++)
    {
        vertexNormal_Unsafe(mesh, V1, Ti[i], vertexNormals[i]);
        vertexNormalLengths[i] = norm_L2_Static<double, 3>(vertexNormals[i]);
        scaleVectorIP_Static<double, 3>(1.0 / vertexNormalLengths[i], vertexNormals[i]);
    }

    double u[3] = { u_[0], u_[1], 1.0 - u_[0] - u_[1] };

    double normal[3];
    makeTriInterpolatedVector_Static<double, 3>(u, vertexNormals[0], vertexNormals[1], vertexNormals[2], normal);
    normalizeVector_Static<double, 3>(normal);

    double weightedLengths = (u[0] * vertexNormalLengths[0] +
                              u[1] * vertexNormalLengths[1] +
                              u[2] * vertexNormalLengths[2]);

    e[0] = w * weightedLengths * (SN[0] - normal[0]);
    e[1] = w * weightedLengths * (SN[1] - normal[1]);
    e[2] = w * weightedLengths * (- normal[2]);
}

void silhouetteNormalResiduals2Jac_V1_Unsafe(const Mesh & mesh, const Matrix<double> & V1, 
                                             int faceIndex, const double * u_, 
                                             const double * SN,
                                             int vertexIndex, const double & w, double * J)
{
    // current triangle and full barycentric coordinates
    const int * Ti = mesh.GetTriangle(faceIndex);
    double u[3] = { u_[0], u_[1], 1.0 - u_[0] - u_[1] };

    // get one rings for all vertices in the face
    std::vector<int> oneRings[3];
    for (int i=0; i<3; i++)
        oneRings[i] = mesh.GetNRing(Ti[i], 1, true);

    // construct set of all vertices involved in the normal calculation
    std::set<int> allVertices;
    for (int i=0; i<3; i++)
        allVertices.insert(oneRings[i].begin(), oneRings[i].end());

    // get the index of the given vertex 
    auto j = allVertices.find(vertexIndex);
    if (j == allVertices.end())
    {
        // vertex index is not in the extended one ring so has no effect
        fillVector_Static<double, 9>(0., J);
        return;
    }

    // build an index mapping for `allVertices` (which is ordered)
    std::map<int, int> indexAllVertices;
    auto it = allVertices.begin();
    for (int l=0; it != allVertices.end(); l++, it++)
        indexAllVertices.insert(std::pair<int, int>(*it, l));

    // construct the un-normalised normal estimate jacobian for **all** vertices in the 
    // extended one-ring
    double normalisedVertexNormals[3][3];
    Matrix<double> unJ(3, 3*allVertices.size(), 0.);

    Matrix<double> wJ(1, 3*allVertices.size(), 0.);
    double weightedLengths = 0.;

    for (int i = 0; i < 3; i++)
    {
        // get the (un-normalised) vertex normal
        double * vertexNormal = normalisedVertexNormals[i];
        vertexNormal_Unsafe(mesh, V1, Ti[i], vertexNormal);

        // get the normalisation jacobian
        Matrix<double> normalizationJac(3,3);
        normalizationJac_Unsafe(vertexNormal, normalizationJac[0]);

        // get the length jacobian
        Matrix<double> lengthJac(1,3);
        lengthJac_Unsafe(vertexNormal, lengthJac[0]);

        // get the Jacobian for the given vertex (column ordering same as `oneRings[i]`)
        Matrix<double> unVertexNormalJ = vertexNormalJac(mesh, V1, Ti[i]);

        // apply the normalisation Jacobian (chain rule)
        Matrix<double> vertexJ(3, unVertexNormalJ.num_cols());
        multiply_A_B(normalizationJac, unVertexNormalJ, vertexJ);

        // apply the length Jacobian (chain rule)
        Matrix<double> lengthJ(1, unVertexNormalJ.num_cols());
        multiply_A_B(lengthJac, unVertexNormalJ, lengthJ);

        // add and scale the columns into the main Jacobian matrices
        auto it = oneRings[i].begin();
        for (int l=0; it != oneRings[i].end(); it++, l++)
        {
            auto p = indexAllVertices.find(*it);
            assert(p != indexAllVertices.end());
            int m = p->second;

            for (int k=0; k<3; k++)
            {
                for (int r=0; r<3; r++)
                    unJ[r][3*m + k] += u[i] * vertexJ[r][3*l + k];

                wJ[0][3*m + k] += u[i] * lengthJ[0][3*l + k];
            }

        }

        // get the vertex normal length
        double vertexNormalLength = norm_L2_Static<double, 3>(vertexNormal);
        weightedLengths += u[i] * vertexNormalLength;

        // normalise the vertex normal
        scaleVectorIP_Static<double, 3>(1.0 / vertexNormalLength, vertexNormal);
    }

    // construct blended (unnormalised) normal vector
    double normal[3];
    makeTriInterpolatedVector_Static<double, 3>(u, 
            normalisedVertexNormals[0],  
            normalisedVertexNormals[1], 
            normalisedVertexNormals[2], normal);

    // construct the final normalisation jacobian
    Matrix<double> normalizationJac(3,3);
    normalizationJac_Unsafe(normal, normalizationJac[0]);

    // apply to the Jacobian for p2 (chain rule)
    Matrix<double> Jp2(3, unJ.num_cols());
    multiply_A_B(normalizationJac, unJ, Jp2);
    scaleMatrixIP(-1.0, Jp2);

    // construct the normal vector
    normalizeVector_Static<double, 3>(normal);

    // construct the residual
    double residual[3];
    residual[0] = SN[0] - normal[0];
    residual[1] = SN[1] - normal[1];
    residual[2] =  - normal[2];

    // construct the final Jacobian
    Matrix<double> finalJ(3, unJ.num_cols());
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < unJ.num_cols(); c++)
            finalJ[r][c] = w * (wJ[0][c] * residual[r] + Jp2[r][c] * weightedLengths);

    // return slice for the vertex of interest
    auto p = indexAllVertices.find(vertexIndex);
    assert(p != indexAllVertices.end());
    int desiredIndex = p->second;

    Matrix<double> outJ(3, 3, J);
    copyMatrixSlice(finalJ, 0, 3*desiredIndex, 3, 3, outJ, 0, 0);
}


PyObject * EvaluateSilhouetteNormal2(PyArrayObject * npy_T,
                                     PyArrayObject * npy_V,
                                     int faceIndex,
                                     PyArrayObject * npy_u,
                                     PyArrayObject * npy_sn,
                                     double w,
                                     bool debug)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);

    Mesh mesh(V.num_rows(), T); 

    PYARRAY_AS_VECTOR(double, npy_u, u);
    PYARRAY_AS_VECTOR(double, npy_sn, sn);

    if (debug)
        asm("int $0x3");

    npy_intp dim(3);
    PyArrayObject * npy_e = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
    PYARRAY_AS_VECTOR(double, npy_e, e);

    silhouetteNormalResiduals2_Unsafe(mesh, V, faceIndex, &u[0], &sn[0],
                                      w, &e[0]);

    return (PyObject *)npy_e;
}

PyObject * EvaluateSilhouette2Jac_V1(PyArrayObject * npy_T,
                                     PyArrayObject * npy_V,
                                     int faceIndex,
                                     PyArrayObject * npy_u,
                                     PyArrayObject * npy_sn,
                                     int vertexIndex,
                                     double w,
                                     bool debug)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);

    Mesh mesh(V.num_rows(), T); 

    PYARRAY_AS_VECTOR(double, npy_u, u);
    PYARRAY_AS_VECTOR(double, npy_sn, sn);

    if (debug)
        asm("int $0x3");

    npy_intp dim[2] = {3,3};
    PyArrayObject * npy_J = (PyArrayObject *)PyArray_SimpleNew(2, dim, NPY_FLOAT64);
    PYARRAY_AS_MATRIX(double, npy_J, J);

    silhouetteNormalResiduals2Jac_V1_Unsafe(mesh, V, faceIndex, &u[0], &sn[0], vertexIndex, w, J[0]);

    return (PyObject *)npy_J;
}

#endif

