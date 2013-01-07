#ifndef __TEST_ROTATION_COMPOSITION_H__
#define __TEST_ROTATION_COMPOSITION_H__

#include "Solve/node.h"
#include "Math/rotation_composition.h"
#include "Util/pyarray_conversion.h"
#include <iostream>
#include <utility>
using namespace std;

PyObject * EvaluateRotationComposition(PyObject * list_Xb,
                                       PyObject * list_y,
                                       int k,
                                       bool debug)

{
    if (debug)
        asm("int $0x3");

    auto Xb = make_vectorOfMatrix<double>(list_Xb); 
    auto y = make_vectorOfMatrix<double>(list_y); 

    vector<const RotationNode *> nodes_Xb;
    for (int i = 0; i < Xb.size(); i++)
        nodes_Xb.push_back(new RotationNode(Xb[i]));

    vector<const CoefficientsNode *> nodes_y;
    for (int i = 0; i < y.size(); i++)
        nodes_y.push_back(new CoefficientsNode(y[i]));

    RotationComposition composition(nodes_Xb, nodes_y);

    PyObject * py_ret = PyTuple_New(2* Xb.size() + 1);

    npy_intp dim(3);
    PyArrayObject * npy_x = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
    PYARRAY_AS_VECTOR(double, npy_x, x);

    composition.Rotation_Unsafe(k, &x[0]);

    PyTuple_SetItem(py_ret, 0, (PyObject *)npy_x);
    npy_intp jacDims[2][2] = {{3, 3}, {3, 1}};

    for (int i = 0; i < Xb.size(); i++)
    {
        for (int j = 0; j < 2; j++)
        {
            PyArrayObject * npy_J = (PyArrayObject *)PyArray_SimpleNew(2, jacDims[j], NPY_FLOAT64);
            PYARRAY_AS_MATRIX(double, npy_J, J);

            composition.Jacobian_Unsafe(i, k, 
                                        j == 0,     // isRotation
                                        J[0]);

            PyTuple_SetItem(py_ret, 2*i + j + 1, (PyObject *)npy_J);
        }
    }

    dealloc_vector(nodes_Xb);
    dealloc_vector(nodes_y);

    return py_ret;
}


#endif
