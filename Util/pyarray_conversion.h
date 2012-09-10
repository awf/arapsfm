#ifndef __PYARRAY_CONVERSION_H__
#define __PYARRAY_CONVERSION_H__

// Requires inclusion into a Python header / Cython file

// Includes
#include <cassert>

#include <Math/v3d_linear.h>
using namespace V3D;

// PyArray_SafeCast
template <typename Elem>
inline
Elem * PyArray_SafeCast(PyArrayObject * arr)
{
    return static_cast<Elem *>(PyArray_DATA(arr));
}

// PyList_to_vector_of_Vector
template <typename Elem>
vector<Vector<Elem> *> PyList_to_vector_of_Vector(PyObject * list)
{
    vector<Vector<Elem> *> vec;

    Py_ssize_t len = PyList_GET_SIZE(list);
    for (Py_ssize_t i=0; i < len; i++)
    {
        PyObject * objA = PyList_GET_ITEM(list, i);
        if (objA == Py_None)
        {
            vec.push_back(nullptr);
            continue;
        }

        PyArrayObject * A = (PyArrayObject *)objA;
        assert(PyArray_NDIM(A) == 1);
        vec.push_back(new Vector<Elem>(PyArray_DIMS(A)[0], PyArray_SafeCast<Elem>(A)));
    }

    return vec;
}

// PyList_to_vector_of_Matrix
template <typename Elem>
vector<Matrix<Elem> *> PyList_to_vector_of_Matrix(PyObject * list)
{
    vector<Matrix<Elem> *> vec;

    Py_ssize_t len = PyList_GET_SIZE(list);
    for (Py_ssize_t i=0; i < len; i++)
    {
        PyObject * objA = PyList_GET_ITEM(list, i);
        if (objA == Py_None)
        {
            vec.push_back(nullptr);
            continue;
        }

        PyArrayObject * A = (PyArrayObject *)objA;
        assert(PyArray_NDIM(A) == 2);
        vec.push_back(new Matrix<Elem>(PyArray_DIMS(A)[0], PyArray_DIMS(A)[1], PyArray_SafeCast<Elem>(A)));
    }

    return vec;
}

// PyList_vector_dealloc
template <typename T>
void PyList_vector_dealloc(vector<T *> & vec)
{
    for (int i=0; i < vec.size(); i++)
    {
        if (vec[i] != nullptr)
        {
            delete vec[i];
            vec[i] = nullptr;
        }
    }

    vec.clear();
}

// Convenience macros
#define PYARRAY_AS_VECTOR(T,A,V) assert(PyArray_NDIM(A) == 1); Vector<T> V(PyArray_DIMS(A)[0], PyArray_SafeCast<T>(A))
#define PYARRAY_AS_MATRIX(T,A,M) assert(PyArray_NDIM(A) == 2); Matrix<T> M(PyArray_DIMS(A)[0], PyArray_DIMS(A)[1], PyArray_SafeCast<T>(A))

#endif
