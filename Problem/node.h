#ifndef __NODE_H__
#define __NODE_H__

#include <memory>
using namespace std;

// Node
class Node
{
    virtual int GetGlobalCount() const { return _globalCount; }
    virtual int GetId() const { return _paramId; }
    virtual int SetId(int paramId) { _paramId = paramId; }

public:
    Node(int count) 
    {
        _globalCount += count;
    }

    static int _globalCount, _paramId;
}
    
// VertexNode
class VertexNode : public Node
{
public:
    typedef shared_ptr<VertexNode> Pointer;

    static VertexNode::Pointer New(int count)
    {
        return VertexNode::Pointer(new VertexNode(count));
    }

    const double * GetVertex(int i) const
    {
        return (*_v)[i];
    }

    ~VertexNode()
    {
        if (_ownsData) delete _v;
    }

protected:
    VertexNode(int count) 
        : Node(count)
    {
        _v = new Matrix<int>(count, 3);
        _ownsData = true;
    }

    VertexNode(Matrix<int> & v)
        : Node(v.num_rows())
    {
        _v = &v;
        _ownsData = false;
    }

    Matrix<double> * _v;
};

#endif
