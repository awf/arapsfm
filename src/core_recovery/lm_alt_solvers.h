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

// Regular instance-core alternation (with free rotations solved for in both)

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
                   bool fixedScale,
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

    auto nodes = new ScaleNode(s);
    nodes->SetPreconditioner(preconditioners[2]);
    if (fixedScale)
        problem.AddFixedNode(nodes);
    else
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
    problem.AddEnergy(new RigidTransformARAPEnergy3(
        *nodeV, *nodeXg, *nodes, *nodeX, *nodeV1, 
        mesh, sqrt(lambdas[0]), uniformWeights, fixedScale));

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

    delete residualTransform;

    return ret;
}

// solve_core
int solve_core(PyArrayObject * npy_T,
               PyArrayObject * npy_V,
               PyObject * list_Xg,
               PyObject * list_s,
               PyObject * list_X,
               PyObject * list_V1,
               // PyObject * list_C,
               // PyObject * list_P,
               PyArrayObject * npy_lambdas,
               PyArrayObject * npy_preconditioners,
               int narrowBand,
               bool uniformWeights,
               const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);

    auto Xg = PyList_to_vector_of_Matrix<double>(list_Xg);
    auto s = PyList_to_vector_of_Matrix<double>(list_s);
    auto X = PyList_to_vector_of_Matrix<double>(list_X);
    auto V1 = PyList_to_vector_of_Matrix<double>(list_V1);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    Problem problem;
    Mesh mesh(V.num_rows(), T);

    auto nodeV = new VertexNode(V);
    nodeV->SetPreconditioner(preconditioners[0]);
    problem.AddNode(nodeV);

    vector<GlobalRotationNode *> instGlobalRotationNodes;
    for (auto i = Xg.begin(); i != Xg.end(); ++i)
    {
        instGlobalRotationNodes.push_back(new GlobalRotationNode(*(*i)));
        problem.AddNode(instGlobalRotationNodes.back());
    }
    instGlobalRotationNodes.back()->SetPreconditioner(preconditioners[3]);

    vector<ScaleNode *> instScaleNodes;
    for (int i=0; i < s.size(); ++i)
    {
        instScaleNodes.push_back(new ScaleNode(*s[i]));

        if (i == 0)
            problem.AddFixedNode(instScaleNodes.back());
        else
            problem.AddNode(instScaleNodes.back());
    }
    instScaleNodes.back()->SetPreconditioner(preconditioners[2]);

    vector<RotationNode *> instRotationNodes;
    for (auto i = X.begin(); i != X.end(); ++i)
    {
        instRotationNodes.push_back(new RotationNode(*(*i)));
        problem.AddNode(instRotationNodes.back());
    }
    instRotationNodes.back()->SetPreconditioner(preconditioners[1]);

    vector<VertexNode *> instVertexNodes;
    for (auto i = V1.begin(); i != V1.end(); ++i)
    {
        instVertexNodes.push_back(new VertexNode(*(*i)));
        problem.AddFixedNode(instVertexNodes.back());
    }
    instVertexNodes.back()->SetPreconditioner(preconditioners[0]);

    // RigidTransformARAPEnergy2B
    for (int i = 0; i < instVertexNodes.size(); ++i)
    {
        problem.AddEnergy(new RigidTransformARAPEnergy3B(
            *nodeV, 
            *instGlobalRotationNodes[i], 
            *instScaleNodes[i], 
            *instRotationNodes[i], 
            *instVertexNodes[i], 
            mesh, 
            sqrt(lambdas[0]), 
            uniformWeights,
            i == 0));
    }

    // LaplacianEnergy
    problem.AddEnergy(new LaplacianEnergy(*nodeV, mesh, sqrt(lambdas[1])));

    // Minimise
    int ret = problem.Minimise(*options);

    // dealloc 
    dealloc_vector(Xg);
    dealloc_vector(s);
    dealloc_vector(X);
    dealloc_vector(V1);

    return ret;
}

// solve_forward_sectioned_arap_proj
int solve_forward_sectioned_arap_proj(PyArrayObject * npy_T,
                                      PyArrayObject * npy_V,
                                      PyArrayObject * npy_Xg,
                                      PyArrayObject * npy_s,
                                      PyArrayObject * npy_Xb,
                                      PyArrayObject * npy_y,
                                      PyArrayObject * npy_X,
                                      PyArrayObject * npy_V1,
                                      PyArrayObject * npy_K,
                                      PyArrayObject * npy_C,
                                      PyArrayObject * npy_P,
                                      PyArrayObject * npy_lambdas,
                                      PyArrayObject * npy_preconditioners,
                                      bool isProjection,
                                      bool uniformWeights,
                                      bool fixedScale,
                                      const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    PYARRAY_AS_MATRIX(double, npy_s, s);
    PYARRAY_AS_MATRIX(double, npy_Xb, Xb);
    PYARRAY_AS_MATRIX(double, npy_y, y);
    PYARRAY_AS_MATRIX(double, npy_X, X);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_MATRIX(int, npy_K, K);

    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    Mesh mesh(V.num_rows(), T);

    Problem problem;
    auto nodeV = new VertexNode(V);
    nodeV->SetPreconditioner(preconditioners[0]);
    problem.AddFixedNode(nodeV);

    auto nodeXg = new GlobalRotationNode(Xg);
    nodeXg->SetPreconditioner(preconditioners[2]);
    problem.AddNode(nodeXg);

    auto nodes = new ScaleNode(s);
    nodes->SetPreconditioner(preconditioners[4]);

    if (fixedScale)
        problem.AddFixedNode(nodes);
    else
        problem.AddNode(nodes);

    auto nodeXb = new RotationNode(Xb);
    nodeXb->SetPreconditioner(preconditioners[1]);
    problem.AddNode(nodeXb);

    auto nodey = new CoefficientsNode(y);
    nodey->SetPreconditioner(preconditioners[3]);
    problem.AddNode(nodey);

    auto nodeX = new RotationNode(X);
    problem.AddNode(nodeX);

    auto nodeV1= new VertexNode(V1);
    problem.AddNode(nodeV1);

    problem.AddEnergy(new SectionedBasisArapEnergy(
        *nodeV, *nodeXg, *nodes,
        *nodeXb, *nodey,
        *nodeX, *nodeV1,
        K, mesh, sqrt(lambdas[0]), 
        uniformWeights, 
        false,  // fixedXb
        true,   // fixedV
        false,  // fixedV1
        fixedScale));

    if (isProjection)
        problem.AddEnergy(new ProjectionEnergy(*nodeV1, C, P, sqrt(lambdas[1])));
    else
        problem.AddEnergy(new AbsolutePositionEnergy(*nodeV1, C, P, sqrt(lambdas[1])));

    int status = problem.Minimise(*options);

    return status;
}

// solve_instance_sectioned_arap
int solve_instance_sectioned_arap(PyArrayObject * npy_T,
                                  PyArrayObject * npy_V,
                                  PyArrayObject * npy_Xg,
                                  PyArrayObject * npy_s,
                                  PyArrayObject * npy_K,
                                  PyArrayObject * npy_Xb,
                                  PyArrayObject * npy_y,
                                  PyArrayObject * npy_X,
                                  PyArrayObject * npy_V1,
                                  PyArrayObject * npy_U,
                                  PyArrayObject * npy_L,
                                  PyArrayObject * npy_S,
                                  PyArrayObject * npy_SN,
                                  PyArrayObject * npy_Rx,
                                  PyArrayObject * npy_Ry,
                                  PyArrayObject * npy_C,
                                  PyArrayObject * npy_P,
                                  PyArrayObject * npy_lambdas,
                                  PyArrayObject * npy_preconditioners,
                                  PyArrayObject * npy_piecewisePolynomial,
                                  int narrowBand,
                                  bool uniformWeights,
                                  bool fixedScale,
                                  const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    PYARRAY_AS_MATRIX(double, npy_s, s);

    PYARRAY_AS_MATRIX(int, npy_K, K);
    PYARRAY_AS_MATRIX(double, npy_Xb, Xb);
    PYARRAY_AS_MATRIX(double, npy_y, y);
    PYARRAY_AS_MATRIX(double, npy_X, X);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);

    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);

    PYARRAY_AS_MATRIX(double, npy_S, S);
    PYARRAY_AS_MATRIX(double, npy_SN, SN);
    PYARRAY_AS_MATRIX(double, npy_Rx, Rx);
    PYARRAY_AS_MATRIX(double, npy_Ry, Ry);

    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);

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

    auto nodes = new ScaleNode(s);
    nodes->SetPreconditioner(preconditioners[2]);
    if (fixedScale)
        problem.AddFixedNode(nodes);
    else
        problem.AddNode(nodes);

    auto nodeXb = new RotationNode(Xb);
    problem.AddFixedNode(nodeXb);

    auto nodey = new CoefficientsNode(y);
    nodey->SetPreconditioner(preconditioners[5]);
    problem.AddNode(nodey);

    auto nodeX = new RotationNode(X);
    nodeX->SetPreconditioner(preconditioners[1]);
    problem.AddNode(nodeX);

    auto nodeV1 = new VertexNode(V1);
    problem.AddNode(nodeV1);

    MeshWalker meshWalker(mesh, V1);
    auto nodeU = new BarycentricNode(U, L, meshWalker);
    nodeU->SetPreconditioner(preconditioners[3]);
    problem.AddNode(nodeU);

    // SectionedBasisArapEnergy
    problem.AddEnergy(new SectionedBasisArapEnergy(
        *nodeV, *nodeXg, *nodes,
        *nodeXb, *nodey,
        *nodeX, *nodeV1,
        K, mesh, sqrt(lambdas[0]), 
        uniformWeights, 
        true,   // fixedXb
        true,   // fixedV
        false,  // fixedV1
        fixedScale));

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
    problem.AddEnergy(new ProjectionEnergy(*nodeV1, C, P, sqrt(lambdas[4])));

    // Minimise
    int ret = problem.Minimise(*options);

    delete residualTransform;

    return ret;
}

// solve_core_sectioned_arap
int solve_core_sectioned_arap(PyArrayObject * npy_T,
                              PyArrayObject * npy_V,
                              PyObject * list_Xg,
                              PyObject * list_s,
                              PyArrayObject * npy_K,
                              PyArrayObject * npy_Xb,
                              PyObject * list_y,
                              PyObject * list_X,
                              PyObject * list_V1,
                              // PyObject * list_C,
                              // PyObject * list_P,
                              PyArrayObject * npy_lambdas,
                              PyArrayObject * npy_preconditioners,
                              int narrowBand,
                              bool uniformWeights,
                              const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);

    PYARRAY_AS_MATRIX(int, npy_K, K);
    PYARRAY_AS_MATRIX(double, npy_Xb, Xb);

    auto Xg = PyList_to_vector_of_Matrix<double>(list_Xg);
    auto s = PyList_to_vector_of_Matrix<double>(list_s);
    auto y = PyList_to_vector_of_Matrix<double>(list_y);
    auto X = PyList_to_vector_of_Matrix<double>(list_X);
    auto V1 = PyList_to_vector_of_Matrix<double>(list_V1);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    Problem problem;
    Mesh mesh(V.num_rows(), T);

    auto nodeV = new VertexNode(V);
    nodeV->SetPreconditioner(preconditioners[0]);
    problem.AddNode(nodeV);

    vector<GlobalRotationNode *> instGlobalRotationNodes;
    for (auto i = Xg.begin(); i != Xg.end(); ++i)
    {
        instGlobalRotationNodes.push_back(new GlobalRotationNode(*(*i)));
        problem.AddNode(instGlobalRotationNodes.back());
    }
    instGlobalRotationNodes.back()->SetPreconditioner(preconditioners[3]);

    vector<ScaleNode *> instScaleNodes;
    for (int i=0; i < s.size(); ++i)
    {
        instScaleNodes.push_back(new ScaleNode(*s[i]));

        if (i == 0)
            problem.AddFixedNode(instScaleNodes.back());
        else
            problem.AddNode(instScaleNodes.back());
    }
    instScaleNodes.back()->SetPreconditioner(preconditioners[2]);

    auto nodeXb = new RotationNode(Xb);
    problem.AddNode(nodeXb);
    nodeXb->SetPreconditioner(preconditioners[1]);

    vector<CoefficientsNode *> instBasisCoefficientNodes;
    for (auto i = y.begin(); i != y.end(); ++i)
    {
        instBasisCoefficientNodes.push_back(new CoefficientsNode(*(*i)));
        problem.AddNode(instBasisCoefficientNodes.back());
    }
    instBasisCoefficientNodes.back()->SetPreconditioner(preconditioners[4]);

    vector<RotationNode *> instRotationNodes;
    for (auto i = X.begin(); i != X.end(); ++i)
    {
        instRotationNodes.push_back(new RotationNode(*(*i)));
        problem.AddNode(instRotationNodes.back());
    }

    vector<VertexNode *> instVertexNodes;
    for (auto i = V1.begin(); i != V1.end(); ++i)
    {
        instVertexNodes.push_back(new VertexNode(*(*i)));
        problem.AddFixedNode(instVertexNodes.back());
    }

    // SectionedBasisArapEnergy
    for (int i = 0; i < instVertexNodes.size(); ++i)
    {
        problem.AddEnergy(new SectionedBasisArapEnergy(
            *nodeV, *instGlobalRotationNodes[i], *instScaleNodes[i], 
            *nodeXb, *instBasisCoefficientNodes[i],
            *instRotationNodes[i], *instVertexNodes[i], 
            K, mesh, sqrt(lambdas[0]), 
            uniformWeights,
            false,  // fixedXb   
            false,  // fixedV
            true,   // fixedV1
            i == 0  // fixedScale
            ));
    }

    // GlobalRotationsDifferenceEnergy
    for (int i = 1; i < instVertexNodes.size(); ++i)
    {
        problem.AddEnergy(new GlobalRotationsDifferenceEnergy(
            *instGlobalRotationNodes[i], 
            *instGlobalRotationNodes[i-1],
            sqrt(lambdas[1]),
            false // fixed0
            ));
    }

    // LaplacianEnergy
    problem.AddEnergy(new LaplacianEnergy(*nodeV, mesh, sqrt(lambdas[2])));

    // Minimise
    int ret = problem.Minimise(*options);

    // dealloc 
    dealloc_vector(Xg);
    dealloc_vector(s);
    dealloc_vector(y);
    dealloc_vector(X);
    dealloc_vector(V1);

    return ret;
}

// solve_instance_sectioned_arap_temporal
int solve_instance_sectioned_arap_temporal(PyArrayObject * npy_T,
                                           PyArrayObject * npy_V,
                                           PyArrayObject * npy_Xg,
                                           PyArrayObject * npy_s,
                                           PyArrayObject * npy_K,
                                           PyArrayObject * npy_Xb,
                                           PyArrayObject * npy_y,
                                           PyArrayObject * npy_X,
                                           PyArrayObject * npy_V1,
                                           PyArrayObject * npy_U,
                                           PyArrayObject * npy_L,
                                           PyArrayObject * npy_S,
                                           PyArrayObject * npy_SN,
                                           PyArrayObject * npy_Rx,
                                           PyArrayObject * npy_Ry,
                                           PyArrayObject * npy_C,
                                           PyArrayObject * npy_P,
                                           PyArrayObject * npy_Vp,
                                           PyArrayObject * npy_Xgp,
                                           PyArrayObject * npy_sp,
                                           PyArrayObject * npy_Xp,
                                           PyArrayObject * npy_lambdas,
                                           PyArrayObject * npy_preconditioners,
                                           PyArrayObject * npy_piecewisePolynomial,
                                           int narrowBand,
                                           bool uniformWeights,
                                           bool fixedScale,
                                           const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    PYARRAY_AS_MATRIX(double, npy_s, s);

    PYARRAY_AS_MATRIX(int, npy_K, K);
    PYARRAY_AS_MATRIX(double, npy_Xb, Xb);
    PYARRAY_AS_MATRIX(double, npy_y, y);
    PYARRAY_AS_MATRIX(double, npy_X, X);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);

    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);

    PYARRAY_AS_MATRIX(double, npy_S, S);
    PYARRAY_AS_MATRIX(double, npy_SN, SN);
    PYARRAY_AS_MATRIX(double, npy_Rx, Rx);
    PYARRAY_AS_MATRIX(double, npy_Ry, Ry);

    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);
    
    PYARRAY_AS_MATRIX(double, npy_Vp, Vp);
    PYARRAY_AS_MATRIX(double, npy_Xgp, Xgp);
    PYARRAY_AS_MATRIX(double, npy_sp, sp);
    PYARRAY_AS_MATRIX(double, npy_Xp, Xp);

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

    auto nodes = new ScaleNode(s);
    nodes->SetPreconditioner(preconditioners[2]);
    if (fixedScale)
        problem.AddFixedNode(nodes);
    else
        problem.AddNode(nodes);

    auto nodeXb = new RotationNode(Xb);
    problem.AddFixedNode(nodeXb);

    auto nodey = new CoefficientsNode(y);
    nodey->SetPreconditioner(preconditioners[5]);
    problem.AddNode(nodey);

    auto nodeX = new RotationNode(X);
    nodeX->SetPreconditioner(preconditioners[1]);
    problem.AddNode(nodeX);

    auto nodeV1 = new VertexNode(V1);
    problem.AddNode(nodeV1);

    MeshWalker meshWalker(mesh, V1);
    auto nodeU = new BarycentricNode(U, L, meshWalker);
    nodeU->SetPreconditioner(preconditioners[3]);
    problem.AddNode(nodeU);

    auto nodeVp = new VertexNode(Vp);
    problem.AddFixedNode(nodeVp);

    auto nodeXgp = new GlobalRotationNode(Xgp);
    problem.AddNode(nodeXgp);

    auto nodesp = new ScaleNode(sp);
    problem.AddNode(nodesp);

    auto nodeXp = new RotationNode(Xp);
    problem.AddNode(nodeXp);

    // SectionedBasisArapEnergy
    problem.AddEnergy(new SectionedBasisArapEnergy(
        *nodeV, *nodeXg, *nodes,
        *nodeXb, *nodey,
        *nodeX, *nodeV1,
        K, mesh, sqrt(lambdas[0]), 
        uniformWeights, 
        true,   // fixedXb
        true,   // fixedV
        false,  // fixedV1
        fixedScale));

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
    problem.AddEnergy(new ProjectionEnergy(*nodeV1, C, P, sqrt(lambdas[4])));

    if (Xp.num_rows() > 0)
    {
        problem.AddEnergy(new RigidTransformARAPEnergy3(
            *nodeVp, *nodeXgp, *nodesp, *nodeXp, *nodeV1,
            mesh, sqrt(lambdas[5]), uniformWeights, false));

        problem.AddEnergy(new GlobalRotationRegulariseEnergy(*nodeXgp, sqrt(Xp.num_rows() * lambdas[6])));
        problem.AddEnergy(new GlobalScaleRegulariseEnergy(*nodesp, sqrt(Xp.num_rows() * lambdas[7])));
        problem.AddEnergy(new RotationRegulariseEnergy(*nodeXp, sqrt(lambdas[8])));
    }

    // Minimise
    int ret = problem.Minimise(*options);

    delete residualTransform;

    return ret;
}

// solve_two_source_arap_proj
int solve_two_source_arap_proj(PyArrayObject * npy_T,
                               PyArrayObject * npy_V,
                               PyArrayObject * npy_Xg,
                               PyArrayObject * npy_s,
                               PyArrayObject * npy_X,
                               PyArrayObject * npy_Vp,
                               PyArrayObject * npy_Xgp,
                               PyArrayObject * npy_sp,
                               PyArrayObject * npy_Xp,
                               PyArrayObject * npy_V1,
                               PyArrayObject * npy_C,
                               PyArrayObject * npy_P,
                               PyArrayObject * npy_lambdas,
                               PyArrayObject * npy_preconditioners,
                               bool uniformWeights,
                               const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    PYARRAY_AS_MATRIX(double, npy_s, s);
    PYARRAY_AS_MATRIX(double, npy_X, X);
    PYARRAY_AS_MATRIX(double, npy_Vp, Vp);
    PYARRAY_AS_MATRIX(double, npy_Xgp, Xgp);
    PYARRAY_AS_MATRIX(double, npy_sp, sp);
    PYARRAY_AS_MATRIX(double, npy_Xp, Xp);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);
    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    Mesh mesh(V.num_rows(), T);

    auto nodeV = new VertexNode(V);
    nodeV->SetPreconditioner(preconditioners[0]);

    auto nodeXg = new GlobalRotationNode(Xg);
    nodeXg->SetPreconditioner(preconditioners[3]);

    auto nodes = new ScaleNode(s);
    nodes->SetPreconditioner(preconditioners[2]);

    auto nodeX = new RotationNode(X);
    nodeX->SetPreconditioner(preconditioners[1]);

    auto nodeVp = new VertexNode(Vp);
    auto nodeXgp = new GlobalRotationNode(Xgp);
    auto nodesp = new ScaleNode(sp);
    auto nodeXp = new RotationNode(Xp);

    auto nodeV1 = new VertexNode(V1);

    Problem problem;
    problem.AddFixedNode(nodeV);
    problem.AddNode(nodeXg);
    problem.AddNode(nodes);
    problem.AddNode(nodeX);
    problem.AddFixedNode(nodeVp);
    problem.AddNode(nodeXgp);
    problem.AddNode(nodesp);
    problem.AddNode(nodeXp);
    problem.AddNode(nodeV1);

    problem.AddEnergy(new ProjectionEnergy(*nodeV1, C, P, sqrt(lambdas[0])));
    problem.AddEnergy(new RigidTransformARAPEnergy3(
        *nodeV, *nodeXg, *nodes, *nodeX, *nodeV1, 
        mesh, sqrt(lambdas[1]), uniformWeights, false));

    if (Xp.num_rows() > 0)
    {
        problem.AddEnergy(new RigidTransformARAPEnergy3(
            *nodeVp, *nodeXgp, *nodesp, *nodeXp, *nodeV1,
            mesh, sqrt(lambdas[2]), uniformWeights, false));

        problem.AddEnergy(new GlobalRotationRegulariseEnergy(*nodeXgp, sqrt(Xp.num_rows() * lambdas[3])));
        problem.AddEnergy(new GlobalScaleRegulariseEnergy(*nodesp, sqrt(Xp.num_rows() * lambdas[4])));
        problem.AddEnergy(new RotationRegulariseEnergy(*nodeXp, sqrt(lambdas[5])));
    }

    return problem.Minimise(*options);
}

#endif
