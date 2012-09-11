#ifndef __TEST_PROBLEM_H__
#define __TEST_PROBLEM_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Energy/arap.h"
#include "Energy/projection.h"
#include "Energy/narrow_band_silhouette.h"
#include "Energy/laplacian.h"
#include "Solve/node.h"
#include "Solve/problem.h"
#include "Solve/optimiser_options.h"

#include "Util/pyarray_conversion.h"

// test_problem
int test_problem(PyArrayObject * npy_V,
                 PyArrayObject * npy_T,
                 PyArrayObject * npy_X,
                 PyArrayObject * npy_V1,
                 PyArrayObject * npy_C,
                 PyArrayObject * npy_P,
                 PyArrayObject * npy_lambdas,
                 const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_X, X);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);
    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    Mesh mesh(V.num_rows(), T);

    VertexNode * nodeV = new VertexNode(V);
    VertexNode * nodeV1 = new VertexNode(V1);
    RotationNode * nodeX = new RotationNode(X);

    Problem problem;
    problem.AddFixedNode(nodeV);
    problem.AddNode(nodeV1);
    problem.AddNode(nodeX);

    ARAPEnergy * arapEnergy = new ARAPEnergy(*nodeV, *nodeX, *nodeV1, mesh, sqrt(lambdas[0]));
    ProjectionEnergy * projEnergy = new ProjectionEnergy(*nodeV1, C, P, sqrt(lambdas[1]));

    problem.AddEnergy(arapEnergy);
    problem.AddEnergy(projEnergy);

    return problem.Minimise(*options);
}

// test_problem2
int test_problem2(PyArrayObject * npy_V,
                  PyArrayObject * npy_T,
                  PyArrayObject * npy_C,
                  PyArrayObject * npy_P,
                  PyArrayObject * npy_lambdas,
                  const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);
    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    Mesh mesh(V.num_rows(), T);

    VertexNode * nodeV = new VertexNode(V);

    Problem problem;
    problem.AddNode(nodeV);

    LaplacianEnergy * lapEnergy = new LaplacianEnergy(*nodeV, mesh, sqrt(lambdas[0]));
    ProjectionEnergy * projEnergy = new ProjectionEnergy(*nodeV, C, P, sqrt(lambdas[1]));

    problem.AddEnergy(lapEnergy);
    problem.AddEnergy(projEnergy);

    return problem.Minimise(*options);
}

// test_problem3
int test_problem3(PyArrayObject * npy_V,
                  PyArrayObject * npy_T,
                  PyArrayObject * npy_U,
                  PyArrayObject * npy_L,
                  PyArrayObject * npy_S,
                  PyArrayObject * npy_SN,
                  PyArrayObject * npy_lambdas,
                  PyArrayObject * npy_preconditioners,
                  int narrowBand,
                  const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);

    PYARRAY_AS_MATRIX(double, npy_S, S);
    PYARRAY_AS_MATRIX(double, npy_SN, SN);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    Mesh mesh(V.num_rows(), T);

    VertexNode * nodeV = new VertexNode(V);
    nodeV->SetPreconditioner(preconditioners[0]);

    MeshWalker meshWalker(mesh, V);
    BarycentricNode * nodeU = new BarycentricNode(U, L, meshWalker);
    nodeU->SetPreconditioner(preconditioners[1]);

    Problem problem;
    problem.AddNode(nodeV);
    problem.AddNode(nodeU);

    LaplacianEnergy * lapEnergy = new LaplacianEnergy(*nodeV, mesh, sqrt(lambdas[0]));
    SilhouetteProjectionEnergy * silProjEnergy = new SilhouetteProjectionEnergy(*nodeV, *nodeU, S, mesh,
        sqrt(lambdas[1]), narrowBand);

    problem.AddEnergy(lapEnergy);
    problem.AddEnergy(silProjEnergy);

    return problem.Minimise(*options);
}

#endif




