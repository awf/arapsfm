#ifndef __NODE_H__
#define __NODE_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include <iostream>
using namespace std;

#include "Geometry/mesh_walker.h"

// Node
class Node
{
public:
    virtual ~Node() {}

    virtual int TypeId() const = 0;
    virtual int Dimension() const = 0;

    virtual void SetParamId(int paramId) { _paramId = paramId; }
    virtual int GetParamId() const { return _paramId; }

    // virtual void SetCount(int count) { _count = count; }
    virtual int GetCount() const { return _count; }

    virtual void SetOffset(int offset) { _offset = offset; }
    virtual int GetOffset() const { return _offset; }

    virtual const double & GetPreconditioner () const = 0;

    virtual void Update(const VectorArrayAdapter<double> & delta)
    {
        for (int i=0; i < _X.num_rows(); i++)
            for (int j=0; j < _X.num_cols(); j++)
                _X[i][j] += delta[i][j];
    }

    virtual void Save() { copyMatrix(_X, _savedX); }
    virtual void Restore() { copyMatrix(_savedX, _X); }
    
    virtual double SquareLength() const
    {
        const Vector<double> flatX(_X.num_rows() * _X.num_cols(), _X[0]);
        return sqrNorm_L2(flatX);
    }

protected:
    Node(Matrix<double> & X, int offset = 0, int paramId = -1)
        : _X(X), _savedX(X.num_rows(), X.num_cols()),
          _count(X.num_rows()), 
          _offset(offset),
          _paramId(paramId)
    {
       this->Save(); 
    }

    Matrix<double> & _X;
    Matrix<double> _savedX;

    int _count;
    int _offset;
    int _paramId;
};

// VertexNode
class VertexNode : public Node
{
public:
    VertexNode(Matrix<double> & V)
        : Node(V)
    {}

    virtual ~VertexNode()
    {}

    virtual const double * GetVertex(int i) const { return _X[i]; }
    virtual const Matrix<double> & GetVertices() const { return _X; }

    virtual int TypeId() const { return 0; }
    virtual int Dimension() const { return 3; }

    virtual void SetPreconditioner(const double & preconditioner) { _preconditioner = preconditioner; }
    virtual const double & GetPreconditioner () const { return _preconditioner; }

protected:
    static double _preconditioner;
};

// RotationNode
class RotationNode : public Node
{
public:
    RotationNode(Matrix<double> & X)
        : Node(X)
    {}

    virtual ~RotationNode()
    {}

    virtual const double * GetRotation(int i) const { return _X[i]; }
    virtual const Matrix<double> & GetRotations() const { return _X; }

    virtual int TypeId() const { return 1; }
    virtual int Dimension() const { return 3; }

    virtual void SetPreconditioner(const double & preconditioner) { _preconditioner = preconditioner; }
    virtual const double & GetPreconditioner () const { return _preconditioner; }

protected:
    static double _preconditioner;
};

// GlobalRotationNode
class GlobalRotationNode : public Node
{
public:
    GlobalRotationNode(Matrix<double> & X)
        : Node(X)
    {}

    virtual ~GlobalRotationNode()
    {}

    virtual const double * GetRotation() const { return _X[0]; }

    virtual int TypeId() const { return 6; }
    virtual int Dimension() const { return 3; }

    virtual void SetPreconditioner(const double & preconditioner) { _preconditioner = preconditioner; }
    virtual const double & GetPreconditioner () const { return _preconditioner; }

protected:
    static double _preconditioner;
};

// BarycentricNode
class BarycentricNode : public Node
{
public:
    BarycentricNode(Matrix<double> & U, Vector<int> & L, const MeshWalker & meshWalker)
        : Node(U), _L(L), _savedL(L.size()),
         _meshWalker(meshWalker)
    {}

    virtual int TypeId() const { return 2; }
    virtual int Dimension() const { return 2; }

    virtual const double * GetBarycentricCoordinate(int i) const { return _X[i]; }
    virtual const Matrix<double> & GetBarycentricCoordinates() const { return _X; }

    virtual int GetFaceIndex(int i) const { return _L[i]; }
    virtual const Vector<int> & GetFaceIndices() const { return _L; }

    virtual void Update(const VectorArrayAdapter<double> & delta)
    {
        _meshWalker.applyDisplacement(_X, _L, delta);
    }

    virtual void Save()
    {
        Node::Save();
        copyVector(_L, _savedL);
    }

    virtual void Restore()
    {
        Node::Restore();
        copyVector(_savedL, _L);
    }

    virtual void SetPreconditioner(const double & preconditioner) { _preconditioner = preconditioner; }
    virtual const double & GetPreconditioner () const { return _preconditioner; }
        
protected:
    Vector<int> & _L;
    Vector<int> _savedL;
    const MeshWalker & _meshWalker;

    static double _preconditioner;
};

// LengthAdjustedBarycentricNode
class LengthAdjustedBarycentricNode : public BarycentricNode
{
public:
    LengthAdjustedBarycentricNode(Matrix<double> & U, Vector<int> & L, const MeshWalker & meshWalker)
        : BarycentricNode(U, L, meshWalker), 
          _Q(U.num_rows(), U.num_cols()),
          _savedQ(U.num_rows(), U.num_cols())
    {
        UpdateInternalLengths();
    }

    virtual int TypeId() const { return 3; }

    virtual const double * GetLengthAdjustedBarycentricCoordinate(int i) const { return _Q[i]; }
    virtual const Matrix<double> & GetLengthAdjustedBarycentricCoordinates() const { return _Q; }

    virtual void Update(const VectorArrayAdapter<double> & delta)
    {
        const Mesh & mesh = _meshWalker.getMesh();
        const Matrix<double> & V = _meshWalker.getVertices();

        Vector<double> newDelta_(delta.count() * delta.size());
        VectorArrayAdapter<double> newDelta(delta.count(), delta.size(), newDelta_.begin());

        for (int i = 0; i < delta.count(); i++)
        {
            // get target face
            const int * Ti = mesh.GetTriangle(_L[i]);

            // get face edge vectors
            const double * Vi = V[Ti[0]], * Vj = V[Ti[1]], * Vk = V[Ti[2]];

            double Vik[3], Vjk[3];
            subtractVectors_Static<double, 3>(Vi, Vk, Vik);
            subtractVectors_Static<double, 3>(Vj, Vk, Vjk);

            // set new delta for barycentric coordinates
            newDelta[i][0] = delta[i][0] / norm_L2_Static<double, 3>(Vik);
            newDelta[i][1] = delta[i][1] / norm_L2_Static<double, 3>(Vjk);
        }

        BarycentricNode::Update(newDelta);

        UpdateInternalLengths();
    }

    virtual void Save()
    {
        BarycentricNode::Save();
        copyMatrix(_Q, _savedQ);
    }

    virtual void Restore()
    {
        BarycentricNode::Restore();
        copyMatrix(_savedQ, _Q);
    }

    virtual void SetPreconditioner(const double & preconditioner) { _preconditioner = preconditioner; }
    virtual const double & GetPreconditioner () const { return _preconditioner; }

protected:
    void UpdateInternalLengths()
    {
        const Mesh & mesh = _meshWalker.getMesh();
        const Matrix<double> & V = _meshWalker.getVertices();

        // convert barycentric coordinates to lengths along the edges which
        // make up a triangle
        for (int i = 0; i < _L.size(); i++)
        {
            const int * Ti = mesh.GetTriangle(_L[i]);
            const double * Vi = V[Ti[0]], * Vj = V[Ti[1]], * Vk = V[Ti[2]];

            double Vik[3], Vjk[3];
            subtractVectors_Static<double, 3>(Vi, Vk, Vik);
            subtractVectors_Static<double, 3>(Vj, Vk, Vjk);

            _Q[i][0] = _X[i][0] * norm_L2_Static<double, 3>(Vik);
            _Q[i][1] = _X[i][1] * norm_L2_Static<double, 3>(Vjk);
        }
    }

    Matrix<double> _Q; 
    Matrix<double> _savedQ;
    static double _preconditioner;
};

// ScaleNode
class ScaleNode : public Node
{
public:
    ScaleNode(Matrix<double> & scale)
        : Node(scale)
    {}

    virtual ~ScaleNode()
    {}

    virtual const double & GetScale() const { return _X[0][0]; }

    virtual int TypeId() const { return 4; }
    virtual int Dimension() const { return 1; }

    virtual void SetPreconditioner(const double & preconditioner) { _preconditioner = preconditioner; }
    virtual const double & GetPreconditioner () const { return _preconditioner; }

    virtual void Update(const VectorArrayAdapter<double> & delta)
    {
        for (int i=0; i < _X.num_rows(); i++)
            for (int j=0; j < _X.num_cols(); j++)
            {
                _X[i][j] += delta[i][j];
                _X[i][j] = _X[i][j] < 0. ? 0. : _X[i][j];
            }
    }

protected:
    static double _preconditioner;
};

// CoefficientsNode
class CoefficientsNode : public Node
{
public:
    CoefficientsNode(Matrix<double> & coefficients)
        : Node(coefficients)
    {}

    virtual ~CoefficientsNode()
    {}

    virtual const double & GetCoefficient(int i) const { return _X[i][0]; }

    virtual int TypeId() const { return 5; }
    virtual int Dimension() const { return 1; }

    virtual void SetPreconditioner(const double & preconditioner) { _preconditioner = preconditioner; }
    virtual const double & GetPreconditioner () const { return _preconditioner; }

protected:
    static double _preconditioner;
};

#endif

