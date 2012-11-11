#ifndef __LM_ALT_SOLVERS__
#define __LM_ALT_SOLVERS__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Energy/residual.h"
#include "Energy/arap.h"
#include "Energy/projection.h"
#include "Energy/narrow_band_silhouette.h"
#include "Energy/laplacian.h"
#include "Energy/spillage.h"
#include "Solve/node.h"
#include "Solve/problem.h"
#include "Solve/optimiser_options.h"

#include "Util/pyarray_conversion.h"
#include <utility>

// solve_instance
int solve_instance(PyArrayObject * npy_T,
                   PyArrayObject * npy_V,
                   PyArrayObject * npy_Xg,
                   PyArrayObject * npy_s,
                   PyArrayObject * npy_X,
                   PyArrayObject * npy_V1,
                   PyArrayObject * npy_U,
                   PyArrayObject * npy_L,
                   // PyArrayObject * npy_C,
                   // PyArrayObject * npy_P,
                   PyArrayObject * npy_S,
                   PyArrayObject * npy_SN,
                   PyArrayObject * npy_Rx,
                   PyArrayObject * npy_Ry,
                   PyArrayObject * npy_lambdas,
                   PyArrayObject * npy_preconditioners,
                   PyArrayObject * npy_piecewisePolynomial,
                   int narrowBand,
                   bool uniformWeights,
                   const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    PYARRAY_AS_MATRIX(double, npy_s, s);
    PYARRAY_AS_MATRIX(double, npy_X, X);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);

    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);

    // PYARRAY_AS_VECTOR(int, npy_C, C);
    // PYARRAY_AS_MATRIX(double, npy_P, P);
    PYARRAY_AS_MATRIX(double, npy_S, S);
    PYARRAY_AS_MATRIX(double, npy_SN, SN);
    PYARRAY_AS_MATRIX(double, npy_Rx, Rx);
    PYARRAY_AS_MATRIX(double, npy_Ry, Ry);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);
    PYARRAY_AS_VECTOR(double, npy_piecewisePolynomial, piecewisePolynomial);

    Problem problem;
    Mesh mesh(V.num_rows(), T);

    auto nodeV = new VertexNode(V);
    nodeV->SetPreconditioner(preconditioners[0]);
    problem.AddFixedNode(nodeV);

    auto nodeXg = new GlobalRotationNode(Xg);
    nodeXg->SetPreconditioner(preconditioners[4]);
    problem.AddNode(nodeXg);

    // Invert scale for `RigidTransformARAPEnergy2`
    s[0][0] = 1.0 / s[0][0];
    auto nodes = new ScaleNode(s);
    nodes->SetPreconditioner(preconditioners[2]);
    problem.AddNode(nodes);

    auto nodeX = new RotationNode(X);
    nodeX->SetPreconditioner(preconditioners[1]);
    problem.AddNode(nodeX);

    auto nodeV1 = new VertexNode(V1);
    problem.AddNode(nodeV1);

    MeshWalker meshWalker(mesh, V1);
    auto nodeU = new BarycentricNode(U, L, meshWalker);
    nodeU->SetPreconditioner(preconditioners[3]);
    problem.AddNode(nodeU);

    // RigidTransformARAPEnergy2
    problem.AddEnergy(new RigidTransformARAPEnergy2(
        *nodeV, *nodeXg, *nodes, *nodeX, *nodeV1, 
        mesh, sqrt(lambdas[0]), uniformWeights));

    // SilhouetteProjectionEnergy
    auto residualTransform = new PiecewisePolynomialTransform_C1(
        piecewisePolynomial[0], piecewisePolynomial[1]);

    problem.AddEnergy(new SilhouetteProjectionEnergy(
        *nodeV1, *nodeU, S, mesh, sqrt(lambdas[1]), narrowBand, residualTransform));

    // SilhouetteNormalEnergy
    problem.AddEnergy(new SilhouetteNormalEnergy(
        *nodeV1, *nodeU, SN, mesh, sqrt(lambdas[2]), narrowBand));

    // Spillage
    problem.AddEnergy(new SpillageEnergy(*nodeV1, Rx, Ry, sqrt(lambdas[3])));

    // Projection
    // problem.AddEnergy(new ProjectionEnergy(*nodeV1, C, P, sqrt(lambdas[4])));

    // Minimise
    int ret = problem.Minimise(*options);

    // Invert scale for `RigidTransformARAPEnergy2`
    s[0][0] = 1.0 / s[0][0];

    delete residualTransform;

    return ret;
}

#endif
