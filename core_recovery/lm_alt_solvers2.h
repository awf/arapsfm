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
#include "Energy/rigid.h"
#include "Solve/node.h"
#include "Solve/problem.h"
#include "Solve/optimiser_options.h"

#include "Util/pyarray_conversion.h"
#include <utility>

// solve_core
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
               bool uniformWeights,
               bool fixedXgb,
               const OptimiserOptions * options,
               PyObject * callback)
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
        problem.AddNode(nodes_s.back());

        nodes_s.back()->SetPreconditioner(preconditioners[2]);
    }

    vector<RotationNode *> nodes_Xgb;
    for (int i = 0; i < Xgb.size(); i++)
    {
        nodes_Xgb.push_back(new RotationNode(*Xgb[i]));
        problem.AddNode(nodes_Xgb.back(), fixedXgb);

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
    node_Xb->SetPreconditioner(preconditioners[1]);

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

    Vector<int> kgLookup(nodes_V1.size()); 
    int j = 0;
    for (int i = 0; i < nodes_V1.size(); i++)
    {
        kgLookup[i] = j;

        if (kg[j] == CompleteSectionedArapEnergy::FIXED_ROTATION)
            j += 1;
        else if (kg[j] == CompleteSectionedArapEnergy::INDEPENDENT_ROTATION)
            j += 2;
        else
            j += 1 + 2 * kg[j];
    }

    // CompleteSectionedArapEnergy
    for (int i = 0; i < nodes_V1.size(); i++)
    {
        vector<const RotationNode *> Xgbi;
        vector<const CoefficientsNode *> ygi;
        const RotationNode * Xgi = nullptr;

        int l = kgLookup[i];
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
            fixedXgb,  // fixedXgb
            false,  // fixedXb  
            false,  // fixedV
            true,   // fixedV1
            false,  // fixedScale
            false   // fixedXg
            ));
    }

    // SectionedArapLinearCombinationEnergy (acceleration)
    for (int i = 1; i < nodes_V1.size() - 1; i++)
    {
        vector<const RotationNode *> X_nodes;
        for (int j = i + 1; j >= i - 1; j--)
            if (j < nodes_X.size())
                X_nodes.push_back(nodes_X[j]);

        vector<const CoefficientsNode *> y_nodes;
        for (int j = i + 1; j >= i - 1; j--)
            if (j < nodes_y.size())
                y_nodes.push_back(nodes_y[j]);

        Vector<double> rotationCoefficients(3);
        rotationCoefficients[0] = 1.0;
        rotationCoefficients[1] = -2.0;
        rotationCoefficients[2] = 1.0;

        Vector<int> fixed(3);
        fillVector(0, fixed);

        problem.AddEnergy(new SectionedArapLinearCombinationEnergy(
            k, *node_Xb,
            move(X_nodes),
            move(y_nodes),
            move(rotationCoefficients),
            sqrt(lambdas[3]),
            move(fixed),
            false));
    }

    // GlobalScalesLinearCombinationEnergy (acceleration)
    for (int i = 1; i < nodes_V1.size() - 1; i++)
    {
        vector<const ScaleNode *> s_nodes;
        s_nodes.push_back(nodes_s[i + 1]);
        s_nodes.push_back(nodes_s[i]);
        s_nodes.push_back(nodes_s[i - 1]);

        Vector<double> scaleCoefficients(3);
        scaleCoefficients[0] = 1.0;
        scaleCoefficients[1] = -2.0;
        scaleCoefficients[2] = 1.0;

        Vector<int> fixedScales(3);
        fixedScales[0] = 0;
        fixedScales[1] = 0;
        fixedScales[2] = 0;

        problem.AddEnergy(new GlobalScalesLinearCombinationEnergy(
            move(s_nodes), 
            move(scaleCoefficients), 
            sqrt(V.num_rows() * lambdas[1]),
            move(fixedScales)));
    }

    // GlobalRotationLinearCombinationEnergy (acceleration)
    for (int i = 1; i < nodes_V1.size() - 1; i++)
    {
        Vector<double> rotationCoefficients(3);
        rotationCoefficients[0] = 1.0;
        rotationCoefficients[1] = -2.0;
        rotationCoefficients[2] = 1.0;

        Vector<int> fixedRotations(3);
        fillVector(0, fixedRotations);

        vector<int> arg_kg;
        vector<vector<const RotationNode *>> arg_Xgb; 
        vector<vector<const CoefficientsNode *>> arg_yg; 
        vector<const RotationNode *> arg_Xg;

        for (int j = -1; j < 2; j++)
        {
            vector<const RotationNode *> arg_nodes_Xgb;
            vector<const CoefficientsNode *> arg_nodes_yg;
            const RotationNode * arg_ptr_Xg = nullptr;

            int l = kgLookup[i - j];
            int n = kg[l++];

            if (n == GlobalRotationLinearCombinationEnergy::FIXED_ROTATION)
            { }
            else if (n == GlobalRotationLinearCombinationEnergy::INDEPENDENT_ROTATION)
            {
                arg_ptr_Xg = nodes_Xg[kg[l++]];
            }
            else
            {
                for (int jj = 0; jj < n; jj++)
                {
                    arg_nodes_Xgb.push_back(nodes_Xgb[kg[l++]]); 
                    arg_nodes_yg.push_back(nodes_yg[kg[l++]]);
                }
            }

            arg_kg.push_back(n);
            arg_Xgb.push_back(move(arg_nodes_Xgb));
            arg_yg.push_back(move(arg_nodes_yg));
            arg_Xg.push_back(arg_ptr_Xg);
        }

        problem.AddEnergy(new GlobalRotationLinearCombinationEnergy(move(arg_kg),
                                                                    move(arg_Xgb),
                                                                    move(arg_yg),
                                                                    move(arg_Xg),
                                                                    move(rotationCoefficients),
                                                                    sqrt(V.num_rows() * lambdas[2]),
                                                                    move(fixedRotations),
                                                                    fixedXgb));
    }

    // GlobalScalesLinearCombinationEnergy (speed)
    for (int i = 1; i < nodes_V1.size(); i++)
    {
        vector<const ScaleNode *> s_nodes;
        s_nodes.push_back(nodes_s[i]);
        s_nodes.push_back(nodes_s[i - 1]);

        Vector<double> scaleCoefficients(2);
        scaleCoefficients[0] = 1.0;
        scaleCoefficients[1] = -1.0;

        Vector<int> fixedScales(2);
        fixedScales[0] = 0;
        fixedScales[1] = 0;

        problem.AddEnergy(new GlobalScalesLinearCombinationEnergy(
            move(s_nodes), 
            move(scaleCoefficients), 
            sqrt(V.num_rows() * lambdas[7]),
            move(fixedScales)));
    }

    // GlobalRotationLinearCombinationEnergy (speed)
    for (int i = 1; i < nodes_V1.size(); i++)
    {
        Vector<double> rotationCoefficients(2);
        rotationCoefficients[0] = 1.0;
        rotationCoefficients[1] = -1.0;

        Vector<int> fixedRotations(2);
        fillVector(0, fixedRotations);

        vector<int> arg_kg;
        vector<vector<const RotationNode *>> arg_Xgb; 
        vector<vector<const CoefficientsNode *>> arg_yg; 
        vector<const RotationNode *> arg_Xg;

        for (int j = 0; j < 2; j++)
        {
            vector<const RotationNode *> arg_nodes_Xgb;
            vector<const CoefficientsNode *> arg_nodes_yg;
            const RotationNode * arg_ptr_Xg = nullptr;

            int l = kgLookup[i - j];
            int n = kg[l++];

            if (n == GlobalRotationLinearCombinationEnergy::FIXED_ROTATION)
            { }
            else if (n == GlobalRotationLinearCombinationEnergy::INDEPENDENT_ROTATION)
            {
                arg_ptr_Xg = nodes_Xg[kg[l++]];
            }
            else
            {
                for (int jj = 0; jj < n; jj++)
                {
                    arg_nodes_Xgb.push_back(nodes_Xgb[kg[l++]]); 
                    arg_nodes_yg.push_back(nodes_yg[kg[l++]]);
                }
            }

            arg_kg.push_back(n);
            arg_Xgb.push_back(move(arg_nodes_Xgb));
            arg_yg.push_back(move(arg_nodes_yg));
            arg_Xg.push_back(arg_ptr_Xg);
        }

        problem.AddEnergy(new GlobalRotationLinearCombinationEnergy(move(arg_kg),
                                                                    move(arg_Xgb),
                                                                    move(arg_yg),
                                                                    move(arg_Xg),
                                                                    move(rotationCoefficients),
                                                                    sqrt(V.num_rows() * lambdas[8]),
                                                                    move(fixedRotations),
                                                                    fixedXgb));
    }

    // SectionedArapLinearCombinationEnergy (acceleration)
    for (int i = 1; i < nodes_V1.size() - 1; i++)
    {
        vector<const RotationNode *> X_nodes;
        for (int j = i; j >= i - 1; j--)
            if (j < nodes_X.size())
                X_nodes.push_back(nodes_X[j]);

        vector<const CoefficientsNode *> y_nodes;
        for (int j = i; j >= i - 1; j--)
            if (j < nodes_y.size())
                y_nodes.push_back(nodes_y[j]);

        Vector<double> rotationCoefficients(2);
        rotationCoefficients[0] = 1.0;
        rotationCoefficients[1] = -1.0;

        Vector<int> fixed(2);
        fillVector(0, fixed);

        problem.AddEnergy(new SectionedArapLinearCombinationEnergy(
            k, *node_Xb,
            move(X_nodes),
            move(y_nodes),
            move(rotationCoefficients),
            sqrt(lambdas[9]),
            move(fixed),
            false));
    }

    vector<const ScaleNode *> s_nodes;
    copy(nodes_s.begin(), nodes_s.end(), back_inserter(s_nodes));
    problem.AddEnergy(new LaplacianEnergy(*node_V, move(s_nodes), mesh, sqrt(lambdas[4])));

    if (callback != Py_None)
        problem.SetCallback(callback);

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

// solve_instance
int solve_instance(PyArrayObject * npy_T,
                   PyArrayObject * npy_V,
                   PyArrayObject * npy_s,
                   PyArrayObject * npy_kg,
                   PyObject * list_Xgb,
                   PyObject * list_yg,
                   PyObject * list_Xg,
                   PyArrayObject * npy_k,
                   PyArrayObject * npy_Xb,
                   PyArrayObject * npy_y,
                   PyArrayObject * npy_X, 
                   PyObject * list_y0,
                   PyObject * list_X0,
                   PyArrayObject * npy_V0,
                   PyArrayObject * npy_sp,
                   PyArrayObject * npy_Xgp,
                   PyArrayObject * npy_Xp,
                   PyObject * list_s0,
                   PyArrayObject * npy_V1, 
                   PyArrayObject * npy_U, 
                   PyArrayObject * npy_L, 
                   PyArrayObject * npy_S, 
                   PyArrayObject * npy_SN, 
                   PyArrayObject * npy_C,
                   PyArrayObject * npy_P,
                   PyArrayObject * npy_lambdas,
                   PyArrayObject * npy_preconditioners,
                   PyArrayObject * npy_piecewisePolynomial,
                   int narrowBand,
                   bool uniformWeights,
                   bool fixedScale,
                   bool fixedGlobalRotation,
                   bool noSilhouetteUpdate,
                   const OptimiserOptions * options,
                   PyObject * callback)
{
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    Mesh mesh(V.num_rows(), T);

    Problem problem;
    auto node_V = new VertexNode(V);
    node_V->SetPreconditioner(preconditioners[0]);
    problem.AddFixedNode(node_V);

    PYARRAY_AS_MATRIX(double, npy_s, s);
    auto node_s = new ScaleNode(s);
    node_s->SetPreconditioner(preconditioners[2]);
    if (fixedScale)
        problem.AddFixedNode(node_s);
    else
        problem.AddNode(node_s);

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

    PYARRAY_AS_VECTOR(int, npy_k, k);

    PYARRAY_AS_MATRIX(double, npy_Xb, Xb);
    auto node_Xb = new RotationNode(Xb);
    problem.AddFixedNode(node_Xb);

    PYARRAY_AS_MATRIX(double, npy_y, y);
    auto node_y = new CoefficientsNode(y);
    problem.AddNode(node_y);

    PYARRAY_AS_MATRIX(double, npy_X, X);
    auto node_X = new RotationNode(X);
    problem.AddNode(node_X);

    PYARRAY_AS_MATRIX(double, npy_V0, V0);

    auto s0 = PyList_to_vector_of_Matrix<double>(list_s0);
    auto y0 = PyList_to_vector_of_Matrix<double>(list_y0);
    auto X0 = PyList_to_vector_of_Matrix<double>(list_X0);

    PYARRAY_AS_MATRIX(double, npy_sp, sp);
    PYARRAY_AS_MATRIX(double, npy_Xgp, Xgp);
    PYARRAY_AS_MATRIX(double, npy_Xp, Xp);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);

    auto node_V1 = new VertexNode(V1);
    problem.AddNode(node_V1);

    MeshWalker meshWalker(mesh, V1);
    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);
    auto node_U = new BarycentricNode(U, L, meshWalker);
    node_U->SetPreconditioner(preconditioners[3]);
    // add `node_U` as "fixed" if (noSilhouetteUpdate == True)
    problem.AddNode(node_U, noSilhouetteUpdate);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    vector<const RotationNode *> Xgbi;
    vector<const CoefficientsNode *> ygi;
    const RotationNode * Xgi = nullptr;

    PYARRAY_AS_VECTOR(int, npy_kg, kg);

    Vector<int> ygAdded(nodes_yg.size());
    Vector<int> XgAdded(nodes_Xg.size());
    fillVector(0, ygAdded);
    fillVector(0, XgAdded);

    int l = 0;
    int n = kg[l++];

    if (n == CompleteSectionedArapEnergy::FIXED_ROTATION ) {}
    else if (n == CompleteSectionedArapEnergy::INDEPENDENT_ROTATION)
    {
        int m = kg[l++];
        Xgi = nodes_Xg[m];
        if (fixedGlobalRotation)
        {
            problem.AddFixedNode(nodes_Xg[m]);
            XgAdded[m] = 2;
        }
        else
        {
            problem.AddNode(nodes_Xg[m]);
            XgAdded[m] = 1;
        }
    }
    else
    {
        for (int j=0; j < n; j++)
        {
            Xgbi.push_back(nodes_Xgb[kg[l++]]);

            int m = kg[l++];
            if (!ygAdded[m])
            {
                if (fixedGlobalRotation)
                {
                    problem.AddFixedNode(nodes_yg[m]);
                    ygAdded[m] = 2;
                }
                else
                {
                    problem.AddNode(nodes_yg[m]);
                    ygAdded[m] = 1;
                }
            }
            ygi.push_back(nodes_yg[m]);
        }
    }

    problem.AddEnergy(new CompleteSectionedArapEnergy(
        *node_V, *node_s,
        n, std::move(Xgbi), std::move(ygi), Xgi,
        k, *node_Xb, *node_y, *node_X,
        *node_V1,
        mesh, sqrt(lambdas[0]),
        uniformWeights,
        true,   // fixedXgb
        true,   // fixedXb
        true,   // fixedV
        false,  // fixedV1
        fixedScale,
        fixedGlobalRotation));

    PYARRAY_AS_VECTOR(double, npy_piecewisePolynomial, piecewisePolynomial);

    auto residualTransform = new PiecewisePolynomialTransform_C1(
        piecewisePolynomial[0], piecewisePolynomial[1]);

    PYARRAY_AS_MATRIX(double, npy_S, S);
    if (!noSilhouetteUpdate)
    {
        auto silhouetteProjectionEnergy = new SilhouetteProjectionEnergy(
            *node_V1, *node_U, S, mesh, sqrt(lambdas[1]), narrowBand, residualTransform);
        problem.AddEnergy(silhouetteProjectionEnergy);
        // meshWalker.addEnergy(silhouetteProjectionEnergy);
    }

    PYARRAY_AS_MATRIX(double, npy_SN, SN);
    if (!noSilhouetteUpdate)
    {
        auto silhouetteNormalEnergy = new SilhouetteNormalEnergy(
            *node_V1, *node_U, SN, mesh, sqrt(lambdas[2]), narrowBand);
        problem.AddEnergy(silhouetteNormalEnergy);
        // meshWalker.addEnergy(silhouetteNormalEnergy);
    }

    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);

    if (P.num_cols() == 2)
        problem.AddEnergy(new ProjectionEnergy(*node_V1, C, P, sqrt(lambdas[4])));
    else if (P.num_cols() == 3)
        problem.AddEnergy(new AbsolutePositionEnergy(*node_V1, C, P, sqrt(lambdas[4])));
    else
        assert(false);

    if (V0.num_rows() > 0)
    {
        auto node_V0 = new VertexNode(V0);
        problem.AddFixedNode(node_V0);

        auto node_sp = new ScaleNode(sp);
        problem.AddNode(node_sp);

        auto node_Xgp = new RotationNode(Xgp);
        problem.AddNode(node_Xgp);

        auto node_Xp = new RotationNode(Xp);
        problem.AddNode(node_Xp);

        problem.AddEnergy(new RigidTransformArapEnergy(
            *node_V0, *node_sp, *node_Xgp, *node_Xp,
            *node_V1,
            mesh, sqrt(lambdas[5]), uniformWeights, false));
    }

    vector<ScaleNode *> nodes_s0;
    for (int i = 0; i < s0.size(); i++)
    {
        nodes_s0.push_back(new ScaleNode(*s0[i]));
        problem.AddFixedNode(nodes_s0.back());
    }

    if (s0.size() > 0)
    {
        vector<const ScaleNode *> s_nodes;
        s_nodes.push_back(node_s);
        s_nodes.push_back(nodes_s0[0]);

        Vector<double> scaleCoefficients(2);
        scaleCoefficients[0] = 1.0;
        scaleCoefficients[1] = -1.0;

        Vector<int> fixedScales(2);
        fixedScales[0] = 0;
        fixedScales[1] = 1;

        problem.AddEnergy(new GlobalScalesLinearCombinationEnergy(
            move(s_nodes), 
            move(scaleCoefficients), 
            sqrt(V0.num_rows() * lambdas[9]),
            move(fixedScales)));
    }

    if (s0.size() > 1)
    {
        vector<const ScaleNode *> s_nodes;
        s_nodes.push_back(node_s);
        s_nodes.push_back(nodes_s0[0]);
        s_nodes.push_back(nodes_s0[1]);

        Vector<double> scaleCoefficients(3);
        scaleCoefficients[0] = 1.0;
        scaleCoefficients[1] = -2.0;
        scaleCoefficients[2] = 1.0;

        Vector<int> fixedScales(3);
        fixedScales[0] = 0;
        fixedScales[1] = 1;
        fixedScales[2] = 1;

        problem.AddEnergy(new GlobalScalesLinearCombinationEnergy(
            move(s_nodes), 
            move(scaleCoefficients), 
            sqrt(V0.num_rows() * lambdas[6]),
            move(fixedScales)));
    }

    if (!fixedGlobalRotation)
    {
        if (s0.size() > 0)
        {
            Vector<double> globalRotationCoefficients(2);
            globalRotationCoefficients[0] = 1.0;
            globalRotationCoefficients[1] = -1.0;

            Vector<int> fixedRotations(2);

            vector<int> arg_kg;
            vector<vector<const RotationNode *>> arg_Xgb; 
            vector<vector<const CoefficientsNode *>> arg_yg; 
            vector<const RotationNode *> arg_Xg;

            int l = 0;
            for (int i = 0; i < globalRotationCoefficients.size(); i++)
            {
                vector<const RotationNode *> arg_nodes_Xgb;
                vector<const CoefficientsNode *> arg_nodes_yg;
                const RotationNode * arg_ptr_Xg = nullptr;

                int isFixed = 1;
                int n = kg[l++];

                if (n == GlobalRotationLinearCombinationEnergy::FIXED_ROTATION)
                { }
                else if (n == GlobalRotationLinearCombinationEnergy::INDEPENDENT_ROTATION)
                {
                    int m = kg[l++];
                    arg_ptr_Xg = nodes_Xg[m];

                    if (!XgAdded[m])
                    {
                        problem.AddFixedNode(nodes_Xg[m]);
                        XgAdded[m] = 2;
                    }

                    isFixed &= XgAdded[m] != 1;
                }
                else
                {
                    for (int j = 0; j < n; j++)
                    {
                        arg_nodes_Xgb.push_back(nodes_Xgb[kg[l++]]); 

                        int m = kg[l++];

                        if (!ygAdded[m])
                        {
                            problem.AddFixedNode(nodes_yg[m]);
                            ygAdded[m] = 2;
                        }

                        isFixed &= ygAdded[m] != 1;

                        arg_nodes_yg.push_back(nodes_yg[m]);
                    }
                }

                fixedRotations[i] = isFixed;

                arg_kg.push_back(n);
                arg_Xgb.push_back(move(arg_nodes_Xgb));
                arg_yg.push_back(move(arg_nodes_yg));
                arg_Xg.push_back(arg_ptr_Xg);
            }

            problem.AddEnergy(new GlobalRotationLinearCombinationEnergy(move(arg_kg),
                                                                        move(arg_Xgb),
                                                                        move(arg_yg),
                                                                        move(arg_Xg),
                                                                        move(globalRotationCoefficients),
                                                                        sqrt(V0.num_rows() * lambdas[10]),
                                                                        move(fixedRotations),
                                                                        true));
        }

        if (s0.size() > 1)
        {
            Vector<double> globalRotationCoefficients(3);
            globalRotationCoefficients[0] = 1.0;
            globalRotationCoefficients[1] = -2.0;
            globalRotationCoefficients[2] = 1.0;

            Vector<int> fixedRotations(3);

            vector<int> arg_kg;
            vector<vector<const RotationNode *>> arg_Xgb; 
            vector<vector<const CoefficientsNode *>> arg_yg; 
            vector<const RotationNode *> arg_Xg;

            int l = 0;
            for (int i = 0; i < globalRotationCoefficients.size(); i++)
            {
                vector<const RotationNode *> arg_nodes_Xgb;
                vector<const CoefficientsNode *> arg_nodes_yg;
                const RotationNode * arg_ptr_Xg = nullptr;

                int isFixed = 1;
                int n = kg[l++];

                if (n == GlobalRotationLinearCombinationEnergy::FIXED_ROTATION)
                { }
                else if (n == GlobalRotationLinearCombinationEnergy::INDEPENDENT_ROTATION)
                {
                    int m = kg[l++];
                    arg_ptr_Xg = nodes_Xg[m];

                    if (!XgAdded[m])
                    {
                        problem.AddFixedNode(nodes_Xg[m]);
                        XgAdded[m] = 2;
                    }

                    isFixed &= XgAdded[m] != 1;
                }
                else
                {
                    for (int j = 0; j < n; j++)
                    {
                        arg_nodes_Xgb.push_back(nodes_Xgb[kg[l++]]); 

                        int m = kg[l++];

                        if (!ygAdded[m])
                        {
                            problem.AddFixedNode(nodes_yg[m]);
                            ygAdded[m] = 2;
                        }

                        isFixed &= ygAdded[m] != 1;

                        arg_nodes_yg.push_back(nodes_yg[m]);
                    }
                }

                fixedRotations[i] = isFixed;

                arg_kg.push_back(n);
                arg_Xgb.push_back(move(arg_nodes_Xgb));
                arg_yg.push_back(move(arg_nodes_yg));
                arg_Xg.push_back(arg_ptr_Xg);
            }

            problem.AddEnergy(new GlobalRotationLinearCombinationEnergy(move(arg_kg),
                                                                        move(arg_Xgb),
                                                                        move(arg_yg),
                                                                        move(arg_Xg),
                                                                        move(globalRotationCoefficients),
                                                                        sqrt(V0.num_rows() * lambdas[7]),
                                                                        move(fixedRotations),
                                                                        true));
        }
    }

    vector<RotationNode *> nodes_X0;
    for (int i = 0; i < X0.size(); i++)
    {
        nodes_X0.push_back(new RotationNode(*X0[i]));
        problem.AddFixedNode(nodes_X0.back());
    }

    vector<CoefficientsNode *> nodes_y0;
    for (int i = 0; i < y0.size(); i++)
    {
        nodes_y0.push_back(new CoefficientsNode(*y0[i]));
        problem.AddFixedNode(nodes_y0.back());
    }

    if (y0.size() > 0)
    {
        vector<const RotationNode *> X_nodes;
        X_nodes.push_back(node_X);
        X_nodes.push_back(nodes_X0[0]);

        vector<const CoefficientsNode *> y_nodes;
        y_nodes.push_back(node_y);
        y_nodes.push_back(nodes_y0[0]);

        Vector<double> rotationCoefficients(2);
        rotationCoefficients[0] = 1.0;
        rotationCoefficients[1] = -1.0;

        Vector<int> fixed(2);
        fixed[0] = 0;
        fixed[1] = 1;

        problem.AddEnergy(new SectionedArapLinearCombinationEnergy(
            k, *node_Xb,
            move(X_nodes),
            move(y_nodes),
            move(rotationCoefficients),
            sqrt(lambdas[11]),
            move(fixed),
            true));
    }

    if (y0.size() > 1)
    {
        vector<const RotationNode *> X_nodes;
        X_nodes.push_back(node_X);
        for (int i = 0; i < nodes_X0.size(); i++)
            X_nodes.push_back(nodes_X0[i]);

        vector<const CoefficientsNode *> y_nodes;
        y_nodes.push_back(node_y);
        for (int i = 0; i < nodes_y0.size(); i++)
            y_nodes.push_back(nodes_y0[i]);

        Vector<double> rotationCoefficients(3);
        rotationCoefficients[0] = 1.0;
        rotationCoefficients[1] = -2.0;
        rotationCoefficients[2] = 1.0;

        Vector<int> fixed(3);
        fixed[0] = 0;
        fixed[1] = 1;
        fixed[2] = 1;

        problem.AddEnergy(new SectionedArapLinearCombinationEnergy(
            k, *node_Xb,
            move(X_nodes),
            move(y_nodes),
            move(rotationCoefficients),
            sqrt(lambdas[8]),
            move(fixed),
            true));
    }

    if (callback != Py_None)
        problem.SetCallback(callback);

    int ret = problem.Minimise(*options);

    delete residualTransform;

    dealloc_vector(Xgb);
    dealloc_vector(yg);
    dealloc_vector(Xg);
    dealloc_vector(s0);
    dealloc_vector(y0);
    dealloc_vector(X0);

    return ret;
}

#endif
