#ifndef __PYARRAY_CONVERSION_H__
#define __PYARRAY_CONVERSION_H__

// Requires inclusion into a Python header / Cython file

// Includes
#include <cassert>
#include <utility>

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

// dealloc_vector
template <typename T>
void dealloc_vector(vector<T *> & vec)
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

// make_Matrix
template <typename T>
Matrix<T> make_Matrix(unsigned int rows, unsigned int cols, T * data)
{
    unsigned int size = rows * cols;
    if (size > 0)
        return Matrix<T>(rows, cols, data);
    else
        return Matrix<T>(0, 0, nullptr);
}

// make_vectorOfMatrix
template <typename T>
std::vector<Matrix<T>> make_vectorOfMatrix(PyObject * list)
{
    Py_ssize_t len = PyList_GET_SIZE(list);

    std::vector<Matrix<T>> vec;
    vec.reserve(len);

    for (Py_ssize_t i=0; i < len; ++i)
    {
        PyArrayObject * npy_A = (PyArrayObject *)PyList_GET_ITEM(list, i); 
        vec.push_back(make_Matrix(PyArray_DIMS(npy_A)[0], 
                                  PyArray_DIMS(npy_A)[1],
                                  (T *)PyArray_DATA(npy_A)));
    }

    return vec;
}

// make_vectorOfVector
template <typename T>
std::vector<Vector<T>> make_vectorOfVector(PyObject * list)
{
    Py_ssize_t len = PyList_GET_SIZE(list);

    std::vector<Vector<T>> vec;
    vec.reserve(len);

    for (Py_ssize_t i=0; i < len; ++i)
    {
        PyArrayObject * npy_A = (PyArrayObject *)PyList_GET_ITEM(list, i); 
        vec.push_back(Vector<T>(PyArray_DIMS(npy_A)[0], (T *)PyArray_DATA(npy_A)));
    }

    return vec;
}

// Convenience macros
#define PYARRAY_AS_VECTOR(T,A,V) assert(PyArray_NDIM(A) == 1); Vector<T> V(PyArray_DIMS(A)[0], PyArray_SafeCast<T>(A))
#define PYARRAY_AS_MATRIX(T,A,M) assert(PyArray_NDIM(A) == 2); Matrix<T> M = std::move(make_Matrix(PyArray_DIMS(A)[0], PyArray_DIMS(A)[1], PyArray_SafeCast<T>(A)))

#endif
