#ifndef __LM_ALT_SOLVERS_LINEAR_H__
#define __LM_ALT_SOLVERS_LINEAR_H__

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Energy/projection.h"
#include "Energy/narrow_band_silhouette.h"
#include "Energy/linear_deformation.h"
#include "Solve/node.h"
#include "Solve/problem.h"
#include "Solve/optimiser_options.h"

#include "Util/pyarray_conversion.h"
#include <utility>
using namespace std;

// solve_instance
int solve_instance(PyArrayObject * npy_T,
                   PyArrayObject * npy_V,
                   PyArrayObject * npy_s,
                   PyArrayObject * npy_kg,
                   PyObject * list_Xgb,
                   PyObject * list_yg,
                   PyObject * list_Xg,
                   PyArrayObject * npy_dg,
                   /* TODO Temporal */
                   PyArrayObject * npy_V1,
                   PyArrayObject * npy_U, 
                   PyArrayObject * npy_L, 
                   PyArrayObject * npy_S, 
                   PyArrayObject * npy_SN, 
                   PyArrayObject * npy_C,
                   PyArrayObject * npy_P,
                   PyArrayObject * npy_lambdas,
                   PyArrayObject * npy_preconditioners,
                   int narrowBand,
                   bool uniformWeights,
                   bool fixedScale,
                   bool fixedGlobalRotation,
                   bool fixedTranslation,
                   bool noSilhouetteUpdate,
                   const OptimiserOptions * options,
                   PyObject * callback)
{
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    Mesh mesh(V.num_rows(), T);

    Problem problem;
    // V
    auto node_V = new VertexNode(V);
    node_V->SetPreconditioner(preconditioners[0]);
    problem.AddFixedNode(node_V);

    // s
    PYARRAY_AS_MATRIX(double, npy_s, s);
    auto node_s = new ScaleNode(s);
    node_s->SetPreconditioner(preconditioners[2]);
    problem.AddNode(node_s, fixedScale);

    // Xg
    auto Xgb = PyList_to_vector_of_Matrix<double>(list_Xgb);

    vector<RotationNode *> nodes_Xgb;
    for (int i = 0; i < Xgb.size(); i++)
    {
        nodes_Xgb.push_back(new RotationNode(*Xgb[i]));
        problem.AddFixedNode(nodes_Xgb.back());

        nodes_Xgb.back()->SetPreconditioner(preconditioners[1]);
    }

    auto yg = PyList_to_vector_of_Matrix<double>(list_yg);

    vector<CoefficientsNode *> nodes_yg;
    for (int i = 0; i < yg.size(); i++)
    {
        nodes_yg.push_back(new CoefficientsNode(*yg[i]));
        // problem.AddNode(nodes_yg.back());
        nodes_yg.back()->SetPreconditioner(preconditioners[4]);
    }

    auto Xg = PyList_to_vector_of_Matrix<double>(list_Xg);

    vector<RotationNode *> nodes_Xg;
    for (int i = 0; i < Xg.size(); i++)
    {
        nodes_Xg.push_back(new RotationNode(*Xg[i]));
        // problem.AddNode(nodes_Xg.back());
        nodes_Xg.back()->SetPreconditioner(preconditioners[1]);
    }

    // dg
    PYARRAY_AS_MATRIX(double, npy_dg, dg);
    auto node_dg = new VertexNode(dg);
    node_dg->SetPreconditioner(preconditioners[0]);
    problem.AddNode(node_dg, fixedTranslation);

    // V1
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    auto node_V1 = new VertexNode(V1);
    problem.AddNode(node_V1);

    // U
    MeshWalker meshWalker(mesh, V1);
    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);
    auto node_U = new BarycentricNode(U, L, meshWalker);
    node_U->SetPreconditioner(preconditioners[3]);

    // add `node_U` as "fixed" if noSilhouetteUpdate
    problem.AddNode(node_U, noSilhouetteUpdate);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    vector<const RotationNode *> Xgbi;
    vector<const CoefficientsNode *> ygi;
    const RotationNode * Xgi = nullptr;

    // parse kg[0] and add global rotation nodes as required
    PYARRAY_AS_VECTOR(int, npy_kg, kg);

    Vector<int> ygAdded(nodes_yg.size());
    Vector<int> XgAdded(nodes_Xg.size());
    fillVector(0, ygAdded);
    fillVector(0, XgAdded);

    int l = 0;
    int n = kg[l++];

    if (n == 0) {}
    else if (n == -1)
    {
        int m = kg[l++];
        Xgi = nodes_Xg[m];
        problem.AddNode(nodes_Xg[m], fixedGlobalRotation);
        XgAdded[m] = fixedGlobalRotation ? 2 : 1;
    }
    else
    {
        for (int j=0; j < n; j++)
        {
            // already added as fixed nodes
            Xgbi.push_back(nodes_Xgb[kg[l++]]);

            int m = kg[l++];
            if (!ygAdded[m])
            {
                problem.AddNode(nodes_yg[m], fixedGlobalRotation);
                ygAdded[m] = fixedGlobalRotation ? 2 : 1;
            }
            ygi.push_back(nodes_yg[m]);
        }
    }


    // LinearDeformationEnergy
    problem.AddEnergy(new LinearDeformationEnergy(
        *node_V, *node_s,
        n, std::move(Xgbi), std::move(ygi), Xgi,
        *node_dg,
        *node_V1,
        mesh, sqrt(lambdas[0]),
        uniformWeights,
        true,   // fixedXgb
        true,   // fixedV
        false,  // fixedV1
        fixedScale,
        fixedGlobalRotation,
        fixedTranslation));

    // SilhouetteProjectionEnergy
    PYARRAY_AS_MATRIX(double, npy_S, S);
    if (!noSilhouetteUpdate)
    {
        auto silhouetteProjectionEnergy = new SilhouetteProjectionEnergy(
            *node_V1, *node_U, S, mesh, sqrt(lambdas[1]), narrowBand);
        problem.AddEnergy(silhouetteProjectionEnergy);
        // meshWalker.addEnergy(silhouetteProjectionEnergy);
    }

    // SilhouetteNormalEnergy
    PYARRAY_AS_MATRIX(double, npy_SN, SN);
    if (!noSilhouetteUpdate)
    {
        auto silhouetteNormalEnergy = new SilhouetteNormalEnergy(
            *node_V1, *node_U, SN, mesh, sqrt(lambdas[2]), narrowBand);
        problem.AddEnergy(silhouetteNormalEnergy);
        // meshWalker.addEnergy(silhouetteNormalEnergy);
    }

    // ProjectionEnergy/AbsolutePositionEnergy
    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);

    if (P.num_cols() == 2)
        problem.AddEnergy(new ProjectionEnergy(*node_V1, C, P, sqrt(lambdas[3])));
    else if (P.num_cols() == 3)
        problem.AddEnergy(new AbsolutePositionEnergy(*node_V1, C, P, sqrt(lambdas[3])));
    else
        assert(false);

    if (callback != Py_None)
        problem.SetCallback(callback);

    int ret = problem.Minimise(*options);

    dealloc_vector(Xgb);
    dealloc_vector(yg);
    dealloc_vector(Xg);

    return ret;
}

#endif
