#ifndef __RIGID_H__
#define __RIGID_H__

// Includes
#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/quaternion.h"

// rigidJac_Q_Unsafe
inline void rigidJac_Q_Unsafe(const double * V0, const double w, const double * q, double * Jq)
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

    Jq[0] = w*(-2*x12 - 2*x4);
    Jq[1] = w*(4*x13 - 2*x14 - 2*x3);
    Jq[2] = w*(-2*x6 + 4*x7 + 2*x8);
    Jq[3] = w*(2*x10 - 2*x5);
    Jq[4] = w*(-2*x13 + 4*x14 + 2*x3);
    Jq[5] = w*(-2*x11 - 2*x4);
    Jq[6] = w*(4*x10 - 2*x5 - 2*x9);
    Jq[7] = w*(2*x6 - 2*x7);
    Jq[8] = w*(4*x6 - 2*x7 - 2*x8);
    Jq[9] = w*(-2*x10 + 4*x5 + 2*x9);
    Jq[10] = w*(-2*x11 - 2*x12);
    Jq[11] = w*(2*x13 - 2*x14);
    // END SymPy
}

// rigidJac_X_Unsafe
inline void rigidJac_X_Unsafe(const double * V0, const double w, const double * q, 
                              const double * D, double * J)
{
    double Jq[12];
    rigidJac_Q_Unsafe(V0, w, q, Jq);
    multiply_A_B_Static<double, 3, 4, 3>(Jq, D, J); 
}


// RigidRegistrationEnergy
class RigidRegistrationEnergy : public Energy
{
public:
    RigidRegistrationEnergy(const VertexNode & V0, const VertexNode & V,
                            const ScaleNode & s, const RotationNode & Xg, const VertexNode & d, 
                            const double w)
        : _V0(V0), _V(V), _s(s), _Xg(Xg), _d(d), _w(w)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_s.GetParamId());
        pUsedParamTypes->push_back(_Xg.GetParamId());
        pUsedParamTypes->push_back(_d.GetParamId());

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        switch (i)
        {
        case 0:
            return _s.GetOffset();
        case 1:
            return _Xg.GetOffset();
        case 2:
            return _d.GetOffset();
        default:
            break;
        };

        assert(false);
        return -1;
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _V.GetCount();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        double q[4];
        quat_Unsafe(_Xg.GetRotation(0), q);

        double R[9];
        rotationMatrix_Unsafe(q, R);
        scaleVectorIP_Static<double, 9>(_s.GetScale(), R);

        double RV0[3];
        multiply_A_v_Static<double, 3, 3>(R, _V0.GetVertex(k), RV0);
        addVectors_Static<double, 3>(RV0, _d.GetVertex(0), RV0);

        subtractVectors_Static<double, 3>(_V.GetVertex(k), RV0, &e[0]);
        scaleVectorIP_Static<double, 3>(_w, &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        switch (whichParam)
        {
        case 0:
        {
            // s
            double q[4];
            quat_Unsafe(_Xg.GetRotation(0), q);

            double R[9];
            rotationMatrix_Unsafe(q, R);

            multiply_A_v_Static<double, 3, 3>(R, _V0.GetVertex(k), J[0]);
            scaleVectorIP_Static<double, 3>(-_w, J[0]);
            return;
        }
        case 1:
        {
            // Xg
            double q[4];
            quat_Unsafe(_Xg.GetRotation(0), q);

            double D[12];
            quatDx_Unsafe(_Xg.GetRotation(0), D);

            rigidJac_X_Unsafe(_V0.GetVertex(k), _w * _s.GetScale(), q, D, J[0]);
            return;
        }
        case 2:
        {
            fillVector_Static<double, 9>(0., J[0]);
            J[0][0] = -_w;
            J[1][1] = -_w;
            J[2][2] = -_w;
            return;
        }
        default:
            break;
        }

        assert(false);
    }

protected:
    const VertexNode & _V0;
    const VertexNode & _V;
    const ScaleNode & _s;
    const RotationNode & _Xg;
    const VertexNode & _d;
    const double _w;
};

#endif
