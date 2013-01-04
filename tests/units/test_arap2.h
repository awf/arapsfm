#ifndef __TEST_ARAP2_H__
#define __TEST_ARAP2_H__

#include "Geometry/mesh.h"
#include "Solve/node.h"
#include "Energy/arap2.h"
#include "Util/pyarray_conversion.h"
#include <iostream>
#include <utility>
using namespace std;

PyObject * EvaluateCompleteSectionedBasisArapEnergy(PyArrayObject * npy_T,
                                                    PyArrayObject * npy_V,
                                                    PyArrayObject * npy_s,
                                                    int n, 
                                                    PyObject * list_Xgb,
                                                    PyObject * list_yg,
                                                    PyArrayObject * npy_Xg,
                                                    PyArrayObject * npy_k,
                                                    PyArrayObject * npy_Xb,
                                                    PyArrayObject * npy_y,
                                                    PyArrayObject * npy_X, 
                                                    PyArrayObject * npy_V1, 
                                                    int k_,
                                                    PyArrayObject * npy_jacDims,
                                                    bool debug)
{
    if (debug)
    {
        asm("int $0x3");
    }

    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    VertexNode node_V(V);

    Mesh mesh(V.num_rows(), T); 

    PYARRAY_AS_MATRIX(double, npy_s, s);
    ScaleNode node_s(s);

    auto Xgb = PyList_to_vector_of_Matrix<double>(list_Xgb);
    vector<const RotationNode *> nodes_Xgb;
    for (int i = 0; i < Xgb.size(); i++)
    {
        nodes_Xgb.push_back(new RotationNode(*Xgb[i]));
    }

    auto yg = PyList_to_vector_of_Matrix<double>(list_yg);
    vector<const CoefficientsNode *> nodes_yg;
    for (int i = 0; i < yg.size(); i++)
    {
        nodes_yg.push_back(new CoefficientsNode(*yg[i]));
    }

    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    RotationNode node_Xg(Xg);

    PYARRAY_AS_MATRIX(double, npy_Xb, Xb);
    RotationNode node_Xb(Xb);

    PYARRAY_AS_MATRIX(double, npy_y, y);
    CoefficientsNode node_y(y);

    PYARRAY_AS_MATRIX(double, npy_X, X);
    RotationNode node_X(X);

    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    VertexNode node_V1(V1);

    PYARRAY_AS_VECTOR(int, npy_k, k);

    CompleteSectionedArapEnergy energy(node_V, node_s, 
                                       n, 
                                       vector<const RotationNode *>(nodes_Xgb), 
                                       vector<const CoefficientsNode *>(nodes_yg), 
                                       &node_Xg,
                                       k, node_Xb, node_y, node_X,
                                       node_V1,
                                       mesh, 1.0,
                                       true,
                                       false,
                                       false,
                                       false,
                                       false,
                                       false);

    // Calculate residual
    PyObject * py_list = PyList_New(0);

    npy_intp dim(3);
    PyArrayObject * npy_e = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
    PYARRAY_AS_VECTOR(double, npy_e, e);

    energy.EvaluateResidual(k_, e);

    PyList_Append(py_list, (PyObject *)npy_e);

    // Calculate Jacobians
    PYARRAY_AS_MATRIX(int, npy_jacDims, jacDims);
    
    for (int i = 0; i < jacDims.num_rows(); ++i)
    {
        npy_intp long_jacDims[2] = { static_cast<npy_intp>(jacDims[i][0]),
                                     static_cast<npy_intp>(jacDims[i][1]) };
                    
        PyArrayObject * npy_J = (PyArrayObject *)PyArray_SimpleNew(2, long_jacDims, NPY_FLOAT64);
        PYARRAY_AS_MATRIX(double, npy_J, J);
        energy.EvaluateJacobian(k_, i, J);

        PyList_Append(py_list, (PyObject *)npy_J);
    }

    dealloc_vector(nodes_Xgb);
    dealloc_vector(Xgb);
    dealloc_vector(nodes_yg);
    dealloc_vector(yg);

    return py_list;
}

PyObject * EvaluateGlobalRotationLinearCombinationEnergy(PyArrayObject * npy_kg,
                                                         PyObject * list_Xgb,
                                                         PyObject * list_yg,
                                                         PyObject * list_Xg,
                                                         double w,
                                                         PyArrayObject * npy_A,
                                                         PyArrayObject * npy_fixed,
                                                         bool fixedXgb,
                                                         PyArrayObject * npy_jacDims,
                                                         bool debug)
{
    if (debug)
    {
        asm("int $0x3");
    }

    auto Xgb = make_vectorOfMatrix<double>(list_Xgb);
    auto yg = make_vectorOfMatrix<double>(list_yg);
    auto Xg = make_vectorOfMatrix<double>(list_Xg);

    vector<const RotationNode *> nodes_Xgb;
    for (int i = 0; i < Xgb.size(); i++)
    {
        nodes_Xgb.push_back(new RotationNode(Xgb[i]));
    }

    vector<const CoefficientsNode *> nodes_yg;
    for (int i = 0; i < yg.size(); i++)
    {
        nodes_yg.push_back(new CoefficientsNode(yg[i]));
    }

    vector<const RotationNode *> nodes_Xg;
    for (int i = 0; i < Xg.size(); i++)
    {
        nodes_Xg.push_back(new RotationNode(Xg[i]));
    }

    PYARRAY_AS_VECTOR(int, npy_kg, kg);

    vector<int> arg_kg;
    vector<vector<const RotationNode *>> arg_Xgb;
    vector<vector<const CoefficientsNode *>> arg_yg;
    vector<const RotationNode *> arg_Xg;

    int l = 0;

    PYARRAY_AS_VECTOR(int, npy_fixed, fixed);
    PYARRAY_AS_VECTOR(double, npy_A, A);

    for (int i = 0; i < A.size(); i++)
    {
        vector<const RotationNode *> arg_nodes_Xgb;
        vector<const CoefficientsNode *> arg_nodes_yg;
        const RotationNode * arg_ptr_Xg = nullptr;

        int n = kg[l++];

        if (n == GlobalRotationLinearCombinationEnergy::FIXED_ROTATION)
        { }
        else if (n == GlobalRotationLinearCombinationEnergy::INDEPENDENT_ROTATION)
        {
            arg_ptr_Xg = nodes_Xg[kg[l++]];
        }
        else
        {
            for (int j = 0; j < n; j++)
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

    GlobalRotationLinearCombinationEnergy energy(move(arg_kg),
                                                 move(arg_Xgb),
                                                 move(arg_yg),
                                                 move(arg_Xg),
                                                 move(A),
                                                 w,
                                                 move(fixed),
                                                 fixedXgb);

    PyObject * py_list = PyList_New(0);

    npy_intp dim(3);
    PyArrayObject * npy_e = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
    PYARRAY_AS_VECTOR(double, npy_e, e);

    energy.EvaluateResidual(0, e);

    PyList_Append(py_list, (PyObject *)npy_e);

    // Calculate Jacobians
    PYARRAY_AS_MATRIX(int, npy_jacDims, jacDims);
    
    for (int i = 0; i < jacDims.num_rows(); ++i)
    {
        npy_intp long_jacDims[2] = { static_cast<npy_intp>(jacDims[i][0]),
                                     static_cast<npy_intp>(jacDims[i][1]) };
                    
        PyArrayObject * npy_J = (PyArrayObject *)PyArray_SimpleNew(2, long_jacDims, NPY_FLOAT64);
        PYARRAY_AS_MATRIX(double, npy_J, J);
        energy.EvaluateJacobian(0, i, J);

        PyList_Append(py_list, (PyObject *)npy_J);
    }

    dealloc_vector(nodes_Xgb);
    dealloc_vector(nodes_yg);
    dealloc_vector(nodes_Xg);

    return py_list;
}

#endif

