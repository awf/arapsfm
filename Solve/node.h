#ifndef __NODE_H__
#define __NODE_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include <iostream>
using namespace std;

// Node
class Node
{
public:
    virtual ~Node() {}

    virtual int TypeId() const = 0;
    virtual int Dimension() const = 0;

    virtual void SetParamId(int paramId) { _paramId = paramId; }
    virtual int GetParamId() const { return _paramId; }

    virtual void SetCount(int count) { _count = count; }
    virtual int GetCount() const { return _count; }

    virtual void SetOffset(int offset) { _offset = offset; }
    virtual int GetOffset() const { return _offset; }

    virtual void Update(const VectorArrayAdapter<double> & delta) = 0;
    virtual void Save() = 0;
    virtual void Restore() = 0;

protected:
    Node(int count = 0, int offset = 0, int paramId = -1)
        : _count(count), 
          _offset(offset),
          _paramId(paramId)
    {}

    int _count;
    int _offset;
    int _paramId;
};

// VertexNode
class VertexNode : public Node
{
public:
    VertexNode(Matrix<double> & V)
        : Node(V.num_rows()),
          _V(V),
          _savedV(V.num_rows(), V.num_cols())
    {}

    virtual ~VertexNode()
    {}

    const double * GetVertex(int i) const { return _V[i]; }
    const Matrix<double> & GetVertices() const { return _V; }

    virtual int TypeId() const { return 0; }
    virtual int Dimension() const { return 3; }

    virtual void Update(const VectorArrayAdapter<double> & delta)
    {
        for (int i=0; i < _V.num_rows(); i++)
            for (int j=0; j < 3; j++)
                _V[i][j] += delta[i][j];
    }

    virtual void Save() { copyMatrix(_V, _savedV); }
    virtual void Restore() { copyMatrix(_savedV, _V); }

protected:
    Matrix<double> & _V;
    Matrix<double> _savedV;
};

// RotationNode
class RotationNode : public Node
{
public:
    RotationNode(Matrix<double> & X)
        : Node(X.num_rows()),
          _X(X),
          _savedX(X.num_rows(), X.num_cols())
    {}

    virtual ~RotationNode()
    {}

    const double * GetRotation(int i) const { return _X[i]; }
    const Matrix<double> & GetRotations() const { return _X; }

    virtual int TypeId() const { return 1; }
    virtual int Dimension() const { return 3; }

    virtual void Update(const VectorArrayAdapter<double> & delta)
    {
        for (int i=0; i < _X.num_rows(); i++)
            for (int j=0; j < 3; j++)
                _X[i][j] += delta[i][j];
    }

    virtual void Save() { copyMatrix(_X, _savedX); }
    virtual void Restore() { copyMatrix(_savedX, _X); }

protected:
    Matrix<double> & _X;
    Matrix<double> _savedX;
};

#endif

