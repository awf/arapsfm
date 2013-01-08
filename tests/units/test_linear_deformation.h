#ifndef __TEST_LINEAR_DEFORMATION_H__
#define __TEST_LINEAR_DEFORMATION_H__

#include "Geometry/mesh.h"
#include "Solve/node.h"
#include "Energy/linear_deformation.h"
#include "Util/pyarray_conversion.h"
#include <iostream>
#include <utility>
using namespace std;

PyObject * EvaluateLinearDeformationEnergy(PyArrayObject * npy_T,
                                           PyArrayObject * npy_V,
                                           PyArrayObject * npy_s,
                                           int n, 
                                           PyObject * list_Xgb,
                                           PyObject * list_yg,
                                           PyArrayObject * npy_Xg,
                                           PyArrayObject * npy_dg,
                                           PyArrayObject * npy_V1,
                                           double w,
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

    PYARRAY_AS_MATRIX(double, npy_dg, dg);
    VertexNode node_dg(dg);

    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    VertexNode node_V1(V1);

    LinearDeformationEnergy energy(node_V, node_s,
                                   n, 
                                   vector<const RotationNode *>(nodes_Xgb), 
                                   vector<const CoefficientsNode *>(nodes_yg), 
                                   &node_Xg,
                                   node_dg,
                                   node_V1,
                                   mesh, w,
                                   false,
                                   false,
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

#endif
