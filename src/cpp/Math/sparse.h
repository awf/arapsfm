/* sparse.h */
#ifndef __SPARSE_H__
#define __SPARSE_H__

// Includes
#include "Math/v3d_linear.h"
using namespace V3D;

#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>

// PyCSRInterface
template <typename T>
class PyCSRInterface
{
public:
    // const_iterator
    class const_iterator
    {
    public:
        const_iterator(const Vector<T> & data,
                       const Vector<int> & indptr,
                       const Vector<int> & indices,
                       int l=0)
        : _data(data),
          _indptr(indptr),
          _indices(indices),
          _l(l), 
          _i(0)
        {
            begin(l);
        }

        void begin(int l=0)
        {
            _l = l;

            unsigned int i = 0;
            for (i=0; i < _indptr.size() && _l >= _indptr[i]; i++);
            _i = (int)(i - 1);
        }

        const_iterator & operator++()
        {
            ++_l;
            if (_l >= _indptr[_i+1])
                ++_i;

            return *this;
        }

        const T operator*() const
        {
            // ** UNSAFE **
            return _data[_l];
        }

        int i() const { return _i; }
        int j() const { return _indices[_l]; }

        bool operator==(const const_iterator & rhs) const { return (_l == rhs._l) && (_i == rhs._i); }
        bool operator!=(const const_iterator & rhs) const { return !((*this) == rhs); }

    protected:
        const Vector<T> & _data;
        const Vector<int> & _indptr;
        const Vector<int> & _indices;
        int _l;
        int _i;
    };

    // PyCSRInterface
    PyCSRInterface(PyArrayObject * npy_data,
                   PyArrayObject * npy_indptr,
                   PyArrayObject * npy_indices)
        : _data(PyArray_DIMS(npy_data)[0], (T* )PyArray_DATA(npy_data)),
          _indptr(PyArray_DIMS(npy_indptr)[0], (int *)PyArray_DATA(npy_indptr)),
          _indices(PyArray_DIMS(npy_indices)[0], (int *)PyArray_DATA(npy_indices))
    {}

    const_iterator begin() const { return const_iterator(_data, _indptr, _indices, 0); }
    const_iterator end() const { return const_iterator(_data, _indptr, _indices, (int)_data.size()); }

    const T operator()(int i, int j) const
    {
        if (i < 0 || i >= (int)(_indptr.size() - 1))
            return T(0);

        for (int l = _indptr[i]; l < _indptr[i+1]; l++)
            if (_indices[l] == j)
                return _data[l];

        return T(0);
    }

    void PrintSelf() const
    {
        for (const_iterator it = begin(); it != end(); ++it)
        {
            std::cout << "(" << it.i() << ", " << it.j() << "): " << *it << std::endl;
        }
    }

protected:
    Vector<T> _data;
    Vector<int> _indptr;
    Vector<int> _indices;
};

// PyBlockDiagonalInterface
template <typename T>
class PyBlockDiagonalInterface
{
public:
    // PyBlockDiagonalInterface
    PyBlockDiagonalInterface(PyObject * py_blockList,
                             int rowOffset = 0,
                             int colOffset = 0,
                             T defValue=T(0),
                             bool isSymmetric=false)
        : _defValue(defValue), _isSymmetric(isSymmetric)
    {
        Py_ssize_t i, l = PyList_GET_SIZE(py_blockList);
        _blocks.reserve(l);    

        _rowStarts.newsize(l + 1);
        _rowStarts[0] = rowOffset;

        _colStarts.newsize(l + 1);
        _colStarts[0] = colOffset;

        for (i = 0; i < l; ++i)
        {
            PyArrayObject * npy_block = (PyArrayObject *)PyList_GET_ITEM(py_blockList, i);
            Matrix<T> block(PyArray_DIMS(npy_block)[0],
                            PyArray_DIMS(npy_block)[1],
                            (T *)PyArray_DATA(npy_block));

            _rowStarts[i+1] = _rowStarts[i] + block.num_rows();
            _colStarts[i+1] = _colStarts[i] + block.num_cols();
            _blocks.push_back(std::move(block));
        }

        _blockPtr.newsize(num_rows());

        Py_ssize_t j = 0;
        for (i = -1; i < l; ++i)
            while (j < _rowStarts[i+1])
                _blockPtr[j++] = i;
    }

    int num_rows() const { return _rowStarts[_rowStarts.size() - 1]; }
    int num_cols() const { return _colStarts[_colStarts.size() - 1]; }

    T operator()(int i, int j, bool * hasElement = nullptr) const
    {
        if (hasElement != nullptr)
            *hasElement = false;

        // if symmetric than make sure indexing into the upper blocks
        if (_isSymmetric)
            if (i > j)
                std::swap(i, j);

        if (i < 0 || i >= num_rows())
            return _defValue;

        // find block given by `i`
        int l = _blockPtr[i];

        // if no block then return default value
        if (l < 0 || l >= _blocks.size()) 
            return _defValue;

        // find row `m` and column `n` in the given block
        int n = j - _colStarts[l];
        if (n < 0 || n >= (int)_blocks[l].num_cols()) 
            return _defValue;

        int m = i - _rowStarts[l];
        if (hasElement != nullptr)
            *hasElement = true;

        return _blocks[l][m][n];
    }

protected:
    vector<Matrix<T>> _blocks;
    Vector<int> _rowStarts;
    Vector<int> _colStarts;
    Vector<int> _blockPtr;
    const T _defValue;
    const bool _isSymmetric;
};

template <typename T>
T fromPyArray_1d(PyArrayObject * npy_array, Py_ssize_t l, T defValue=T(0))
{
    if ((PyObject*)npy_array == Py_None || npy_array == nullptr)
        return defValue;

    if (l < 0 || l >= PyArray_DIMS(npy_array)[0])
        return defValue;

    return static_cast<T *>PyArray_DATA(npy_array)[l];
}

// PyMultipleBlockDiagonalInterface
template <typename T>
class PyMultipleBlockDiagonalInterface
{
public:
    PyMultipleBlockDiagonalInterface(PyObject * py_listOfblockList,
                                     PyArrayObject * npy_rowOffsets,
                                     PyArrayObject * npy_colOffsets,
                                     T defValue=T(0),
                                     bool isSymmetric=false)
        : _defValue(defValue)
    {
        Py_ssize_t N = PyList_GET_SIZE(py_listOfblockList); 

        for (Py_ssize_t l = 0; l < N; ++l)
        {
            _blockInterfaces.push_back(new PyBlockDiagonalInterface<T>(
                PyList_GET_ITEM(py_listOfblockList, l),
                fromPyArray_1d<int>(npy_rowOffsets, l),
                fromPyArray_1d<int>(npy_colOffsets, l),
                defValue,
                isSymmetric));
        }
    }

    ~PyMultipleBlockDiagonalInterface()
    {
        for (int i = 0; i < _blockInterfaces.size(); ++i)
            delete _blockInterfaces[i];
    }

    T operator()(int i, int j, bool * hasElement = nullptr) const
    {
        bool hasElement_ = false;
        T ret = _defValue;

        for (int l = 0; !hasElement_ && l < _blockInterfaces.size(); ++l)
        {
            ret = (*_blockInterfaces[l])(i, j, &hasElement_);

            if (hasElement_)
                break;
        }

        if (hasElement != nullptr)
            *hasElement = hasElement_;

        return ret;
    }

protected:
    const T _defValue;
    std::vector<const PyBlockDiagonalInterface<T> *> _blockInterfaces;
};
    
#endif

