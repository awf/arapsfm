#ifndef __LINEAR_BASIS_SHAPE_H__
#define __LINEAR_BASIS_SHAPE_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Geometry/axis_angle.h"
#include "Geometry/quaternion.h"
#include <cassert>

// LinearBasisShapeNode
class LinearBasisShapeNode : public VertexNode
{
public:
    LinearBasisShapeNode(Matrix<double> & V,
                         const vector<const VertexNode *> & Vb,
                         const CoefficientsNode & y,
                         const ScaleNode & s,
                         const RotationNode & Xg,
                         const VertexNode & Vd)
        : VertexNode(V), _V(V),
          _Vb(Vb), _y(y), _s(s), _Xg(Xg), _Vd(Vd),
          _D(y.GetCount()),
          _R(3, 3), _sR(3, 3)
    {
        assert(_Vb.size() == y.GetCount() + 1);
    }

    virtual void Prepare() 
    {
        int N = _Vb[0]->GetVertices().num_rows();
        Matrix<double> W1(N, 3);
        Matrix<double> W2(N, 3);

        copyMatrix(_Vb[0]->GetVertices(), W1);

        for (int i = 0; i < _D; i++)
        {
            scaleMatrix(_Vb[i + 1]->GetVertices(), _y.GetCoefficient(i), W2);
            addMatricesIP(W2, W1);
        }

        double q[4];
        quat_Unsafe(_Xg.GetRotation(0), q);
        rotationMatrix_Unsafe(q, _R[0]);
        scaleMatrix(_R, _s.GetScale(), _sR);

        Matrix<double> sRt(3, 3);
        copyMatrix(_R, sRt);
        transposeMatrixIP(sRt);
        scaleMatrixIP(_s.GetScale(), sRt);

        multiply_A_B(W1, sRt, _V);
        
        const double * d = _Vd.GetVertex(0);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 3; j++)
                _V[i][j] += d[j];
    }

protected:
    Matrix<double> & _V;
    const vector<const VertexNode *> & _Vb;
    const CoefficientsNode & _y;
    const ScaleNode & _s;
    const RotationNode & _Xg;
    const VertexNode & _Vd;
    const int _D;

    Matrix<double> _R, _sR;
};


#endif
