#ifndef __LM_ALT_SOLVERS2_H__
#define __LM_ALT_SOLVERS2_H__

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Energy/residual.h"
#include "Energy/arap2.h"
#include "Energy/projection.h"
#include "Energy/narrow_band_silhouette.h"
#include "Energy/laplacian.h"
#include "Energy/spillage.h"
#include "Solve/node.h"
#include "Solve/problem.h"
#include "Solve/optimiser_options.h"

#include "Util/pyarray_conversion.h"
#include <utility>

// TODO: solve_core: Change number of preconditioners (no GlobalRotation preconditioner now)
int solve_core(PyArrayObject * npy_T,
               PyArrayObject * npy_V,
               PyObject * list_s,
               PyArrayObject * npy_kg,
               PyObject * list_Xgb,
               PyObject * list_yg,
               PyObject * list_Xg,
               PyArrayObject * npy_k,
               PyArrayObject * npy_Xb,
               PyObject * list_y,
               PyObject * list_X,
               PyObject * list_V1,
               PyArrayObject * npy_lambdas,
               PyArrayObject * npy_preconditioners,
               int narrowBand,
               bool uniformWeights,
               const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    auto s = PyList_to_vector_of_Matrix<double>(list_s);

    PYARRAY_AS_VECTOR(int, npy_kg, kg);
    auto Xgb = PyList_to_vector_of_Matrix<double>(list_Xgb);
    auto yg = PyList_to_vector_of_Matrix<double>(list_yg);
    auto Xg = PyList_to_vector_of_Matrix<double>(list_Xg);

    PYARRAY_AS_VECTOR(int, npy_k, k);
    PYARRAY_AS_MATRIX(double, npy_Xb, Xb);
    auto y = PyList_to_vector_of_Matrix<double>(list_y);
    auto X = PyList_to_vector_of_Matrix<double>(list_X);

    auto V1 = PyList_to_vector_of_Matrix<double>(list_V1);

    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);
    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    Problem problem;
    Mesh mesh(V.num_rows(), T);

    auto node_V = new VertexNode(V);
    node_V->SetPreconditioner(preconditioners[0]);
    problem.AddNode(node_V);

    vector<ScaleNode *> nodes_s;
    for (int i = 0; i < s.size(); i++)
    {
        nodes_s.push_back(new ScaleNode(*s[i]));

        if (i == 0)
            problem.AddFixedNode(nodes_s.back());
        else
            problem.AddNode(nodes_s.back());

        nodes_s.back()->SetPreconditioner(preconditioners[2]);
    }

    vector<RotationNode *> nodes_Xgb;
    for (int i = 0; i < Xgb.size(); i++)
    {
        nodes_Xgb.push_back(new RotationNode(*Xgb[i]));
        problem.AddNode(nodes_Xgb.back());

        nodes_Xgb.back()->SetPreconditioner(preconditioners[1]);
    }

    vector<CoefficientsNode *> nodes_yg;
    for (int i = 0; i < yg.size(); i++)
    {
        nodes_yg.push_back(new CoefficientsNode(*yg[i]));
        problem.AddNode(nodes_yg.back());

        nodes_yg.back()->SetPreconditioner(preconditioners[3]);
    }

    vector<RotationNode *> nodes_Xg;
    for (int i = 0; i < Xg.size(); i++)
    {
        nodes_Xg.push_back(new RotationNode(*Xg[i]));
        problem.AddNode(nodes_Xg.back());

        nodes_Xg.back()->SetPreconditioner(preconditioners[1]);
    }

    auto node_Xb = new RotationNode(Xb);
    problem.AddNode(node_Xb);

    vector<CoefficientsNode *> nodes_y;
    for (int i = 0; i < y.size(); i++)
    {
        nodes_y.push_back(new CoefficientsNode(*y[i]));
        problem.AddNode(nodes_y.back());

        nodes_y.back()->SetPreconditioner(preconditioners[3]);
    }

    vector<RotationNode *> nodes_X;
    for (int i = 0; i < X.size(); i++)
    {
        nodes_X.push_back(new RotationNode(*X[i]));
        problem.AddNode(nodes_X.back());

        nodes_X.back()->SetPreconditioner(preconditioners[1]);
    }

    vector<VertexNode *> nodes_V1;
    for (int i = 0; i < V1.size(); i++)
    {
        nodes_V1.push_back(new VertexNode(*V1[i]));
        problem.AddFixedNode(nodes_V1.back());
    }

    int l = 0;
    for (int i = 0; i < nodes_V1.size(); i++)
    {
        vector<const RotationNode *> Xgbi;
        vector<const CoefficientsNode *> ygi;
        const RotationNode * Xgi = nullptr;

        int n = kg[l++];
        if (n == CompleteSectionedArapEnergy::FIXED_ROTATION ) {}
        else if (n == CompleteSectionedArapEnergy::INDEPENDENT_ROTATION)
        {
            Xgi = nodes_Xg[kg[l++]];
        }
        else
        {
            for (int j=0; j < n; j++)
            {
                Xgbi.push_back(nodes_Xgb[kg[l++]]);
                ygi.push_back(nodes_yg[kg[l++]]);
            }
        }

        problem.AddEnergy(new CompleteSectionedArapEnergy(
            *node_V, *nodes_s[i], 
            n, std::move(Xgbi), std::move(ygi), Xgi,
            k, *node_Xb, *nodes_y[i], *nodes_X[i],
            *nodes_V1[i],
            mesh, sqrt(lambdas[0]),
            uniformWeights,
            false,  // fixedXgb
            false,  // fixedXb  
            false,  // fixedV
            true,   // fixedV1
            i == 0   // fixedScale
            ));

        // TODO: Omitted `GlobalRotationsDifferenceEnergy`
    }

    problem.AddEnergy(new LaplacianEnergy(*node_V, mesh, sqrt(lambdas[1])));

    int ret = problem.Minimise(*options);

    dealloc_vector(s);
    dealloc_vector(Xgb);
    dealloc_vector(yg);
    dealloc_vector(Xg);
    dealloc_vector(y);
    dealloc_vector(X);
    dealloc_vector(V1);

    return ret;
}

#endif
