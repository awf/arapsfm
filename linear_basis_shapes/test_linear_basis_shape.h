#ifndef __TEST_LINEAR_BASIS_SHAPE_H__
#define __TEST_LINEAR_BASIS_SHAPE_H__

#include "linear_basis_shape.h"
#include "Util/pyarray_conversion.h"
using namespace std;

PyObject * EvaluateLinearBasisShape(PyObject * list_Vb,
                                    PyArrayObject * npy_y,
                                    PyArrayObject * npy_s,
                                    PyArrayObject * npy_Xg,
                                    PyArrayObject * npy_Vd)
{
    auto Vb = make_vectorOfMatrix<double>(list_Vb);
    PYARRAY_AS_MATRIX(double, npy_y, y);
    PYARRAY_AS_MATRIX(double, npy_s, s);
    PYARRAY_AS_MATRIX(double, npy_Xg, Xg);
    PYARRAY_AS_MATRIX(double, npy_Vd, Vd);

    vector<const VertexNode *> nodes_Vb_const;
    for (int i = 0; i < Vb.size(); i++)
        nodes_Vb_const.push_back(new VertexNode(Vb[i]));

    CoefficientsNode node_y(y);
    ScaleNode node_s(s);
    RotationNode node_Xg(Xg);
    VertexNode node_Vd(Vd);

    npy_intp dim[2] = {Vb[0].num_rows(), Vb[0].num_cols()};
    PyArrayObject * npy_V1 = (PyArrayObject *)PyArray_SimpleNew(2, dim, NPY_FLOAT64);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);

    LinearBasisShapeNode V(V1, nodes_Vb_const, node_y, node_s, node_Xg, node_Vd);
    V.Prepare();

    for (int i = 0; i < Vb.size(); i++)
        delete nodes_Vb_const[i];

    return (PyObject *)npy_V1;
}

#endif
