#ifndef __TEST_PROBLEM_H__
#define __TEST_PROBLEM_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Energy/arap.h"
#include "Energy/projection.h"
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

#endif
