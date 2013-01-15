#ifndef __LBS_LM_SOLVERS_H__
#define __LBS_LM_SOLVERS_H__

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Energy/laplacian.h"

#include "linear_basis_shape.h"
#include "linear_basis_shape_projection.h"
#include "linear_basis_shape_silhouette.h"

#include "Solve/node.h"
#include "Solve/problem.h"
#include "Solve/optimiser_options.h"

#include "Util/pyarray_conversion.h"
#include <vector>
#include <utility>
using namespace std;

// solve_single_projection
int solve_single_projection(PyArrayObject * npy_T,
                            PyObject * list_Vb,
                            PyArrayObject * npy_s,
                            PyArrayObject * npy_Xg,
                            PyArrayObject * npy_Vd,
                            PyArrayObject * npy_y,
                            PyArrayObject * npy_C,
                            PyArrayObject * npy_P,
                            PyArrayObject * npy_lambdas,
                            PyArrayObject * npy_preconditioners,
                            bool debug,
                            const OptimiserOptions * options)
{
    if (debug)
        asm("int $0x3");

    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    PYARRAY_AS_MATRIX(int, npy_T, T);
    auto Vb = make_vectorOfMatrix<double>(list_Vb);
    PYARRAY_AS_MATRIX(double, npy_s, s);
    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    PYARRAY_AS_MATRIX(double, npy_Vd, Vd);
    PYARRAY_AS_MATRIX(double, npy_y, y);
    Mesh mesh(Vb[0].num_rows(), T);

    Problem problem;

    vector<VertexNode *> nodes_Vb;

    for (int i = 0; i < Vb.size(); i++)
    {
        nodes_Vb.push_back(new VertexNode(Vb[i]));
        problem.AddNode(nodes_Vb.back());

        nodes_Vb.back()->SetPreconditioner(preconditioners[0]);
    }

    auto node_s = new ScaleNode(s);
    node_s->SetPreconditioner(preconditioners[2]);
    problem.AddNode(node_s); 

    auto node_Xg = new RotationNode(Xg);
    node_Xg->SetPreconditioner(preconditioners[1]);
    problem.AddNode(node_Xg); 

    auto node_Vd = new VertexNode(Vd);
    problem.AddNode(node_Vd); 

    auto node_y = new CoefficientsNode(y);
    node_y->SetPreconditioner(preconditioners[3]);
    problem.AddNode(node_y); 

    Matrix<double> V_(Vb[0].num_rows(), Vb[0].num_cols());
    auto node_V = new LinearBasisShapeNode(V_, nodes_Vb, *node_y, *node_s, *node_Xg, *node_Vd);
    problem.AddCompositeNode(node_V);

    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    problem.AddEnergy(new LinearBasisShapeProjectionEnergy(*node_V, C, P, 
                      sqrt(lambdas[0])));

    vector<const ScaleNode *> nodes_s;
    nodes_s.push_back(node_s);

    for (int i = 0; i < nodes_Vb.size(); i++)
    {
        problem.AddEnergy(new LaplacianEnergy(*nodes_Vb[i], vector<const ScaleNode *>(nodes_s),
                                              mesh, sqrt(lambdas[1])));
    }

    int ret = problem.Minimise(*options);

    return ret;
}

// solve_single_silhouette
int solve_single_silhouette(PyArrayObject * npy_T,
                            PyObject * list_Vb,
                            PyArrayObject * npy_s,
                            PyArrayObject * npy_Xg,
                            PyArrayObject * npy_Vd,
                            PyArrayObject * npy_y,
                            PyArrayObject * npy_U,
                            PyArrayObject * npy_L,
                            PyArrayObject * npy_C,
                            PyArrayObject * npy_P,
                            PyArrayObject * npy_S,
                            PyArrayObject * npy_SN,
                            PyArrayObject * npy_lambdas,
                            PyArrayObject * npy_preconditioners,
                            int narrowBand,
                            bool debug,
                            const OptimiserOptions * options)
{
    if (debug)
        asm("int $0x3");

    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    PYARRAY_AS_MATRIX(int, npy_T, T);
    auto Vb = make_vectorOfMatrix<double>(list_Vb);
    PYARRAY_AS_MATRIX(double, npy_s, s);
    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    PYARRAY_AS_MATRIX(double, npy_Vd, Vd);
    PYARRAY_AS_MATRIX(double, npy_y, y);
    Mesh mesh(Vb[0].num_rows(), T);

    Problem problem;

    vector<VertexNode *> nodes_Vb;

    for (int i = 0; i < Vb.size(); i++)
    {
        nodes_Vb.push_back(new VertexNode(Vb[i]));
        problem.AddNode(nodes_Vb.back());

        nodes_Vb.back()->SetPreconditioner(preconditioners[0]);
    }

    auto node_s = new ScaleNode(s);
    node_s->SetPreconditioner(preconditioners[2]);
    problem.AddNode(node_s); 

    auto node_Xg = new RotationNode(Xg);
    node_Xg->SetPreconditioner(preconditioners[1]);
    problem.AddNode(node_Xg); 

    auto node_Vd = new VertexNode(Vd);
    problem.AddNode(node_Vd); 

    auto node_y = new CoefficientsNode(y);
    node_y->SetPreconditioner(preconditioners[4]);
    problem.AddNode(node_y); 

    Matrix<double> V_(Vb[0].num_rows(), Vb[0].num_cols());
    auto node_V = new LinearBasisShapeNode(V_, nodes_Vb, *node_y, *node_s, *node_Xg, *node_Vd);
    problem.AddCompositeNode(node_V);

    MeshWalker meshWalker(mesh, V_);
    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);
    auto node_U = new BarycentricNode(U, L, meshWalker);
    node_U->SetPreconditioner(preconditioners[3]);
    problem.AddNode(node_U);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);

    problem.AddEnergy(new LinearBasisShapeProjectionEnergy(*node_V, C, P, 
                      sqrt(lambdas[0])));

    PYARRAY_AS_MATRIX(double, npy_S, S);
    problem.AddEnergy(new LinearBasisShapeSilhouetteProjectionEnergy(*node_V, *node_U,
        S, mesh, sqrt(lambdas[1]), narrowBand));

    PYARRAY_AS_MATRIX(double, npy_SN, SN);
    problem.AddEnergy(new LinearBasisShapeSilhouetteNormalEnergy2(*node_V, *node_U,
        SN, mesh, sqrt(lambdas[2]), narrowBand));

    vector<const ScaleNode *> nodes_s;
    nodes_s.push_back(node_s);

    for (int i = 0; i < nodes_Vb.size(); i++)
    {
        problem.AddEnergy(new LaplacianEnergy(*nodes_Vb[i], vector<const ScaleNode *>(nodes_s),
                                              mesh, sqrt(lambdas[3])));
    }

    if (node_y->GetCount() > 0)
        problem.AddEnergy(new LinearBasisShapeCoefficientEnergy(*node_y, sqrt(lambdas[4])));

    int ret = problem.Minimise(*options);

    return ret;
}

// solve_multiple
int solve_multiple(PyArrayObject * npy_T,
                   PyObject * list_Vb,
                   PyObject * list_s,
                   PyObject * list_Xg,
                   PyObject * list_Vd,
                   PyObject * list_y,
                   PyObject * list_U,
                   PyObject * list_L,
                   PyObject * list_C,
                   PyObject * list_P,
                   PyObject * list_S,
                   PyObject * list_SN,
                   PyArrayObject * npy_lambdas,
                   PyArrayObject * npy_preconditioners,
                   int narrowBand,
                   bool debug,
                   const OptimiserOptions * options)
{
    if (debug)
        asm("int $0x3");

    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    PYARRAY_AS_MATRIX(int, npy_T, T);
    auto Vb = make_vectorOfMatrix<double>(list_Vb);
    auto s = make_vectorOfMatrix<double>(list_s);
    auto Xg = make_vectorOfMatrix<double>(list_Xg);
    auto Vd = make_vectorOfMatrix<double>(list_Vd);
    auto y = make_vectorOfMatrix<double>(list_y);

    auto U = make_vectorOfMatrix<double>(list_U);
    auto L = make_vectorOfVector<int>(list_L);

    assert(s.size() == Xg.size());
    assert(Vd.size() == Xg.size());
    assert(Vd.size() == y.size());
    assert(U.size() == y.size());
    assert(U.size() == L.size());

    Mesh mesh(Vb[0].num_rows(), T);

    Problem problem;

    vector<VertexNode *> nodes_Vb;
    for (int i = 0; i < Vb.size(); i++)
    {
        nodes_Vb.push_back(new VertexNode(Vb[i]));
        problem.AddNode(nodes_Vb.back());

        nodes_Vb.back()->SetPreconditioner(preconditioners[0]);
    }

    vector<ScaleNode *> nodes_s;
    for (int i = 0; i < s.size(); i++)
    {
        nodes_s.push_back(new ScaleNode(s[i]));
        problem.AddNode(nodes_s.back());

        nodes_s.back()->SetPreconditioner(preconditioners[2]);
    }

    vector<RotationNode *> nodes_Xg;
    for (int i = 0; i < Xg.size(); i++)
    {
        nodes_Xg.push_back(new RotationNode(Xg[i]));
        problem.AddNode(nodes_Xg.back());

        nodes_Xg.back()->SetPreconditioner(preconditioners[1]);
    }

    vector<VertexNode *> nodes_Vd;
    for (int i = 0; i < Vd.size(); i++)
    {
        nodes_Vd.push_back(new VertexNode(Vd[i]));
        problem.AddNode(nodes_Vd.back());

        nodes_Vd.back()->SetPreconditioner(preconditioners[0]);
    }

    vector<CoefficientsNode *> nodes_y;
    for (int i = 0; i < y.size(); i++)
    {
        nodes_y.push_back(new CoefficientsNode(y[i]));
        problem.AddNode(nodes_y.back());

        nodes_y.back()->SetPreconditioner(preconditioners[4]);
    }

    vector<Matrix<double> * > V_;
    vector<LinearBasisShapeNode *> nodes_V;
    for (int i = 0; i < y.size(); i++)
    {
        V_.push_back(new Matrix<double>(Vb[0].num_rows(), Vb[0].num_cols()));
        nodes_V.push_back(new LinearBasisShapeNode(*V_.back(), 
                                                   nodes_Vb, 
                                                   *nodes_y[i], 
                                                   *nodes_s[i], 
                                                   *nodes_Xg[i], 
                                                   *nodes_Vd[i]));
        problem.AddCompositeNode(nodes_V.back());
    }

    vector<MeshWalker *> meshWalkers;
    vector<BarycentricNode *> nodes_U;

    for (int i = 0; i < U.size(); i++)
    {
        meshWalkers.push_back(new MeshWalker(mesh, *V_[i]));
        nodes_U.push_back(new BarycentricNode(U[i], L[i], *meshWalkers.back()));
        problem.AddNode(nodes_U.back());

        nodes_U.back()->SetPreconditioner(preconditioners[3]);
    }

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    auto C = make_vectorOfVector<int>(list_C);
    auto P = make_vectorOfMatrix<double>(list_P);

    for (int i = 0; i < C.size(); i++)
    {
        problem.AddEnergy(new LinearBasisShapeProjectionEnergy(*nodes_V[i], C[i], P[i], 
                          sqrt(lambdas[0])));
    }

    auto S = make_vectorOfMatrix<double>(list_S);

    for (int i = 0; i < S.size(); i++)
    {
        problem.AddEnergy(new LinearBasisShapeSilhouetteProjectionEnergy(*nodes_V[i], *nodes_U[i],
            S[i], mesh, sqrt(lambdas[1]), narrowBand));
    }

    auto SN = make_vectorOfMatrix<double>(list_SN);

    for (int i = 0; i < SN.size(); i++)
    {
        problem.AddEnergy(new LinearBasisShapeSilhouetteNormalEnergy2(*nodes_V[i], *nodes_U[i],
            SN[i], mesh, sqrt(lambdas[2]), narrowBand));
    }

    vector<const ScaleNode *> const_nodes_s;
    for (int i = 0; i < nodes_s.size(); i++)
        const_nodes_s.push_back(nodes_s[i]);

    for (int i = 0; i < nodes_Vb.size(); i++)
    {
        problem.AddEnergy(new LaplacianEnergy(*nodes_Vb[i], 
            vector<const ScaleNode *>(const_nodes_s),
            mesh, sqrt(lambdas[3])));
    }

    if (nodes_Vb.size() > 1)
    {
        for (int i = 0; i < nodes_y.size(); i++)
            problem.AddEnergy(new LinearBasisShapeCoefficientEnergy(*nodes_y[i], sqrt(lambdas[4])));
    }

    int ret = problem.Minimise(*options);

    dealloc_vector(meshWalkers);
    dealloc_vector(V_);

    return ret;
}
#endif
