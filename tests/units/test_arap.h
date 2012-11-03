#ifndef __TEST_ARAP_H__
#define __TEST_ARAP_H__

#include "Geometry/mesh.h"
#include "Solve/node.h"
#include "Energy/arap.h"
#include "Util/pyarray_conversion.h"
#include <iostream>
using namespace std;

PyObject * Evaluate(PyArrayObject * npy_T,
                    PyArrayObject * npy_V,
                    PyArrayObject * npy_X,
                    PyArrayObject * npy_Xg,
                    PyArrayObject * npy_s,
                    PyArrayObject * npy_V1,
                    int k,
                    bool verbose)
{
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
    VertexNode node_V(V);

    Mesh mesh(V.num_rows(), T); 

    if (verbose)
    {
        cout << "Evaluate::mesh.GetNumberOfVertices(): " << mesh.GetNumberOfVertices() << endl;
        cout << "Evaluate::mesh.GetNumberOfHalfEdges(): " << mesh.GetNumberOfHalfEdges() << endl;
        cout << "EvaluateResidual::mesh.GetHalfEdge(k, 0): " << mesh.GetHalfEdge(k, 0)  << endl;
        cout << "EvaluateResidual::mesh.GetHalfEdge(k, 1): " << mesh.GetHalfEdge(k, 1)  << endl;
    }

    PYARRAY_AS_MATRIX(double, npy_X, X);
    RotationNode node_X(X);

    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    RotationNode node_Xg(Xg);

    PYARRAY_AS_MATRIX(double, npy_s, s);
    ScaleNode node_s(s);

    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    VertexNode node_V1(V1);

    // Setup `energy`
    RigidTransformARAPEnergy energy(node_V, node_X, node_Xg, node_s, node_V1, mesh, 1.0, true);

    // Calculate residual
    PyObject * py_list = PyList_New(0);

    npy_intp dim(3);
    PyArrayObject * npy_e = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
    PYARRAY_AS_VECTOR(double, npy_e, e);

    energy.EvaluateResidual(k, e);

    PyList_Append(py_list, (PyObject *)npy_e);

    // Calculate Jacobians

    npy_intp dims [][2] = {{3, 3}, {3, 3}, {3, 1}, {3, 3}, {3, 3}};

    for (int i=0; i < 5; i++)
    {
        PyArrayObject * npy_J = (PyArrayObject *)PyArray_SimpleNew(2, dims[i], NPY_FLOAT64);
        PYARRAY_AS_MATRIX(double, npy_J, J);
        energy.EvaluateJacobian(k, i, J);

        PyList_Append(py_list, (PyObject *)npy_J);
    }

    return py_list;
}

#endif
