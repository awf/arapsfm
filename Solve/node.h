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

    virtual int Id() const = 0;
    virtual int Dimension() const = 0;

    virtual void SetCount(int count) { _count = count; }
    virtual void SetOffset(int offset) { _offset = offset; }

    virtual int GetCount() const { return _count; }
    virtual int GetOffset() const { return _offset; }

protected:
    Node(int count = 0, int offset = 0)
        : _count(count), 
          _offset(offset)
    {}

    int _count;
    int _offset;
};

// VertexNode
class VertexNode : public Node
{
public:
    VertexNode(Matrix<double> & v)
        : Node(v.num_rows()),
          _v(v)
    {}

    virtual ~VertexNode()
    {}

    const double * GetVertex(int i) const
    {
        return _v[i];
    }

    virtual int Id() const { return 0; }
    virtual int Dimension() const { return 3; }

protected:
    Matrix<double> & _v;
};

// RotationNode
class RotationNode : public Node
{
public:
    RotationNode(Matrix<double> & x)
        : Node(x.num_rows()),
          _x(x)
    {}

    virtual ~RotationNode()
    {}

    const double * GetRotation(int i) const
    {
        return _x[i];
    }

    virtual int Id() const { return 1; }
    virtual int Dimension() const { return 3; }

protected:
    Matrix<double> & _x;
};

#endif

