#ifndef __LINEAR_BASIS_SHAPE_H__
#define __LINEAR_BASIS_SHAPE_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Geometry/axis_angle.h"
#include "Geometry/quaternion.h"
#include <cassert>

// LinearBasisShapeNode_GlobalRotationJac_Q_Unsafe
inline void LinearBasisShapeNode_GlobalRotationJac_Q_Unsafe(
    const double * V0, const double w, const double * q, double * Jq)
{
    double x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14;

    // BEGIN SymPy
    x0 = V0[2];
    x1 = V0[1];
    x2 = V0[0];
    x3 = q[3]*x0;
    x4 = q[2]*x0;
    x5 = q[1]*x0;
    x6 = q[0]*x0;
    x7 = q[2]*x2;
    x8 = q[3]*x1;
    x9 = q[3]*x2;
    x10 = q[2]*x1;
    x11 = q[0]*x2;
    x12 = q[1]*x1;
    x13 = q[1]*x2;
    x14 = q[0]*x1;

    Jq[0] = -w*(-2*x12 - 2*x4);
    Jq[1] = -w*(4*x13 - 2*x14 - 2*x3);
    Jq[2] = -w*(-2*x6 + 4*x7 + 2*x8);
    Jq[3] = -w*(2*x10 - 2*x5);
    Jq[4] = -w*(-2*x13 + 4*x14 + 2*x3);
    Jq[5] = -w*(-2*x11 - 2*x4);
    Jq[6] = -w*(4*x10 - 2*x5 - 2*x9);
    Jq[7] = -w*(2*x6 - 2*x7);
    Jq[8] = -w*(4*x6 - 2*x7 - 2*x8);
    Jq[9] = -w*(-2*x10 + 4*x5 + 2*x9);
    Jq[10] = -w*(-2*x11 - 2*x12);
    Jq[11] = -w*(2*x13 - 2*x14);
    // END SymPy
}

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
          _R(3, 3), _sR(3, 3), 
          _W1(V.num_rows(), V.num_cols()),
          _W2(V.num_rows(), V.num_cols())
    {
        assert(_Vb.size() == (_D + 1));
    }

    virtual int D() const { return _D; }

    virtual void Prepare() 
    {
        int N = _Vb[0]->GetVertices().num_rows();

        copyMatrix(_Vb[0]->GetVertices(), _W1);

        for (int i = 0; i < _D; i++)
        {
            scaleMatrix(_Vb[i + 1]->GetVertices(), _y.GetCoefficient(i), _W2);
            addMatricesIP(_W2, _W1);
        }

        double q[4];
        quat_Unsafe(_Xg.GetRotation(0), q);
        rotationMatrix_Unsafe(q, _R[0]);
        scaleMatrix(_R, _s.GetScale(), _sR);

        Matrix<double> sRt(3, 3);
        copyMatrix(_R, sRt);
        transposeMatrixIP(sRt);
        scaleMatrixIP(_s.GetScale(), sRt);

        multiply_A_B(_W1, sRt, _V);
        
        const double * d = _Vd.GetVertex(0);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 3; j++)
                _V[i][j] += d[j];
    }

    // `Jacobian` methods
    virtual void VertexJacobian(const int whichParam, const int i, Matrix<double> & J)
    {
        assert(whichParam < (D + 1));

        copyMatrix(_sR, J);
        if (whichParam == 0)
            return;

        scaleMatrixIP(_y.GetCoefficient(whichParam - 1), J);
    }

    virtual void ScaleJacobian(const int i, Matrix<double> & J)
    {
        multiply_A_v_Static<double, 3, 3>(_R[0], _W1[i], J[0]);
    }

    virtual void GlobalRotationJacobian(const int i, Matrix<double> & J)
    {
        double q[4];
        quat_Unsafe(_Xg.GetRotation(0), q);

        double Jq[12];
        LinearBasisShapeNode_GlobalRotationJac_Q_Unsafe(_W1[i], _s.GetScale(), q, Jq);

        double D[12];
        quatDx_Unsafe(_Xg.GetRotation(0), D);

        multiply_A_B_Static<double, 3, 4, 3>(Jq, D, J[0]);
    }

    virtual void DisplacementJacobian(Matrix<double> & J)
    {
        makeIdentityMatrix(J);
    }

    virtual void CoefficientJacobian(const int whichParam, const int i, Matrix<double> & J)
    {
        assert(whichParam < D);
        multiply_A_v_Static<double, 3, 3>(_sR[0], _Vb[whichParam + 1]->GetVertex(i), J[0]);
    }

protected:
    Matrix<double> & _V;
    const vector<const VertexNode *> & _Vb;
    const CoefficientsNode & _y;
    const ScaleNode & _s;
    const RotationNode & _Xg;
    const VertexNode & _Vd;
    const int _D;

    Matrix<double> _R, _sR, _W1, _W2;
};


#endif
