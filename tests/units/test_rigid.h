#ifndef __TEST_RIGID__
#define __TEST_RIGID__

#include "Solve/node.h"
#include "Energy/rigid.h"
#include "Util/pyarray_conversion.h"
#include <iostream>
#include <utility>
using namespace std;

PyObject * EvaluateRigidRegistrationEnergy(PyArrayObject * npy_V0,
                                           PyArrayObject * npy_V,
                                           PyArrayObject * npy_s,
                                           PyArrayObject * npy_xg,
                                           PyArrayObject * npy_d,
                                           double w,
                                           int k,
                                           bool debug)
{
    if (debug)
        asm("int $0x3");

    PYARRAY_AS_MATRIX(double, npy_V0, V0);
    VertexNode node_V0(V0);

    PYARRAY_AS_MATRIX(double, npy_V, V);
    VertexNode node_V(V);

    PYARRAY_AS_MATRIX(double, npy_s, s);
    ScaleNode node_s(s);

    PYARRAY_AS_MATRIX(double, npy_xg, xg);
    RotationNode node_xg(xg);

    PYARRAY_AS_MATRIX(double, npy_d, d);
    VertexNode node_d(d);

    RigidRegistrationEnergy energy(node_V0, node_V, node_s, node_xg, node_d, w);

    PyObject * py_list = PyList_New(0);

    npy_intp dim(3);
    PyArrayObject * npy_e = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
    PYARRAY_AS_VECTOR(double, npy_e, e);

    energy.EvaluateResidual(k, e);
    PyList_Append(py_list, (PyObject *)npy_e);

    npy_intp jacDims[3][2] = { {3, 1}, {3, 3}, {3, 3} };

    for (int i = 0; i < 3; i++)
    {
        PyArrayObject * npy_J = (PyArrayObject *)PyArray_SimpleNew(2, jacDims[i], NPY_FLOAT64);
        PYARRAY_AS_MATRIX(double, npy_J, J);

        energy.EvaluateJacobian(k, i, J);

        PyList_Append(py_list, (PyObject *)npy_J);
    }

    return py_list;
}

#endif
