#ifndef __TEST_PROBLEM_H__
#define __TEST_PROBLEM_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Solve/node.h"
#include "Energy/arap.h"
#include "Solve/problem.h"
#include "Solve/optimiser_options.h"

#include "Util/pyarray_conversion.h"

// test_problem
void test_problem(PyArrayObject * npy_V,
                  PyArrayObject * npy_T,
                  PyArrayObject * npy_X,
                  PyArrayObject * npy_V1,
                  const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_X, X);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);

    Mesh mesh(V.num_rows(), T);

    VertexNode * nodeV = new VertexNode(V);
    VertexNode * nodeV1 = new VertexNode(V1);
    RotationNode * nodeX = new RotationNode(X);

    Problem problem;
    problem.AddFixedNode(nodeV);
    problem.AddNode(nodeV1);
    problem.AddNode(nodeX);

    ARAPEnergy * arapEnergy = new ARAPEnergy(*nodeV, *nodeX, *nodeV1, mesh, 1.0);
    problem.AddEnergy(arapEnergy);

    optimizerVerbosenessLevel = 2;
    int returnStatus = problem.Minimise(*options);
}

#endif
