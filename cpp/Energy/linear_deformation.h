#ifndef __LINEAR_DEFORMATION_H__
#define __LINEAR_DEFORMATION_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Math/rotation_composition.h"
#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/mesh.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <iterator>
using namespace std;

// linearDeformationResiduals_Unsafe
inline void linearDeformationResiduals_Unsafe(
           const double * Vi, const double * V1i,
           const double w, const double * q, 
           const double s, const double * d, double * e)
{
    // BEGIN SymPy
    double x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11;

    x0 = Vi[2];
    x1 = Vi[1];
    x2 = Vi[0];
    x3 = q[2]*q[2];
    x4 = q[2]*q[3];
    x5 = q[0]*q[1];
    x6 = q[0]*q[2];
    x7 = q[1]*q[3];
    x8 = q[1]*q[2];
    x9 = q[0]*q[3];
    x10 = q[1]*q[1];
    x11 = q[0]*q[0];

    e[0] = w*(V1i[0] - d[0] - s*(x0*(2*x6 + 2*x7) + x1*(-2*x4 + 2*x5) + x2*(-2*x10 - 2*x3 + 1)));
    e[1] = w*(V1i[1] - d[1] - s*(x0*(2*x8 - 2*x9) + x1*(-2*x11 - 2*x3 + 1) + x2*(2*x4 + 2*x5)));
    e[2] = w*(V1i[2] - d[2] - s*(x0*(-2*x10 - 2*x11 + 1) + x1*(2*x8 + 2*x9) + x2*(2*x6 - 2*x7)));
    // END SymPy
}

// linearDeformationJac_V_Unsafe
inline void linearDeformationJac_V_Unsafe(const double w, const double * q, const double s, double * J)
{
    double x3, x4, x5, x6, x7, x8, x9, x10, x11;
    x3 = q[2]*q[2];
    x4 = q[2]*q[3];
    x5 = q[0]*q[1];
    x6 = q[0]*q[2];
    x7 = q[1]*q[3];
    x8 = q[1]*q[2];
    x9 = q[0]*q[3];
    x10 = q[1]*q[1];
    x11 = q[0]*q[0];

    J[0] = -2*x10 - 2*x3 + 1;
    J[1] = -2*x4 + 2*x5;
    J[2] = 2*x6 + 2*x7;

    J[3] = 2*x4 + 2*x5;
    J[4] = -2*x11 - 2*x3 + 1;
    J[5] = 2*x8 - 2*x9;

    J[6] = 2*x6 - 2*x7;
    J[7] = 2*x8 + 2*x9;
    J[8] = -2*x10 - 2*x11 + 1;

    scaleVectorIP_Static<double, 9>(s * -w, J);
}

// linearDeformationJac_s_Unsafe
inline void linearDeformationJac_s_Unsafe(const double w, const double * q, 
                                          const double * Vi, double * J)
{
    double R[9];
    rotationMatrix_Unsafe(q, R);

    multiply_A_v_Static<double, 3, 3>(R, Vi, J);
    scaleVectorIP_Static<double, 3>(-w, J);
}

// linearDeformationJac_Q_Unsafe
inline void linearDeformationJac_Q_Unsafe(const double * Vi, const double w, 
                                          const double * q, double * Jq)
{
    double x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14;

    // BEGIN SymPy
    x0 = Vi[2];
    x1 = Vi[1];
    x2 = Vi[0];
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

/*
// linearDeformationJac_X_Unsafe
inline void linearDeformationJac_X_Unsafe(const double * Vi, const double w, 
                                          const double * q, const double * D, 
                                          double * J)
{
    double Jq[12];

    linearDeformationJac_Q_Unsafe(Vi, w, q, Jq);
    
    for (int i=0; i<3; i++)
    {
        double * Ji = J + 3*i;
        const double * Jqi = Jq + 4*i;

        Ji[0] = Jqi[0]*D[0] + Jqi[1]*D[3] + Jqi[2]*D[6] + Jqi[3]*D[9];
        Ji[1] = Jqi[0]*D[1] + Jqi[1]*D[4] + Jqi[2]*D[7] + Jqi[3]*D[10];
        Ji[2] = Jqi[0]*D[2] + Jqi[1]*D[5] + Jqi[2]*D[8] + Jqi[3]*D[11];
    }
}
*/

// LinearDeformationEnergy
class LinearDeformationEnergy : public Energy
{
public:
    LinearDeformationEnergy(const VertexNode & V, 
                            const ScaleNode & s,
                            const int kg, vector<const RotationNode *> && Xgb, vector<const CoefficientsNode *> && yg, const RotationNode * Xg,
                            const VertexNode & dg,
                            const VertexNode & V1,
                            const Mesh & mesh, const double w,
                            bool fixedXgb, bool fixedXb, bool fixedV, bool fixedV1, bool fixedScale, bool fixedXg, bool fixedTranslation)

        : Energy(w), 
          _V(V), _s(s), 
          _kg(kg), _Xgb(Xgb), _yg(yg), _Xg(Xg), _dg(dg), 
          _V1(V1), _mesh(mesh),
          _fixedXgb(fixedXgb), 
          _fixedXb(fixedXb), 
          _fixedV(fixedV), 
          _fixedV1(fixedV1), 
          _fixedScale(fixedScale), 
          _fixedXg(fixedXg), 
          _fixedTranslation(fixedTranslation),
          _XgComp(_Xgb, _yg)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        auto pUsedParamTypes = new vector<int>;

        if (!_fixedV)
            pUsedParamTypes->push_back(_V.GetParamId());

        if (!_fixedScale) 
            pUsedParamTypes->push_back(_s.GetParamId());

        if (!_fixedXg)
        {
            int n = 0;

            switch (_kg)
            {
            case FIXED_ROTATION:
                break;
            case INDEPENDENT_ROTATION:
                pUsedParamTypes->push_back(_Xg->GetParamId());
                break;
            default:
                n = _kg;
                break;
            }

            for (int i=0; i < n; i++)
            {
                if (!_fixedXgb) pUsedParamTypes->push_back(_Xgb[i]->GetParamId());
                pUsedParamTypes->push_back(_yg[i]->GetParamId());
            }
        }

        if (!_fixedTranslation) 
            pUsedParamTypes->push_back(_dg.GetParamId());

        if (!_fixedV1)
            pUsedParamTypes->push_back(_V1.GetParamId());

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int l) const
    {
        int __ARG_COUNT = 0;
        #define IF_PARAM_ON_CONDITION(X) if ((X) && l == __ARG_COUNT++)

        IF_PARAM_ON_CONDITION(!_fixedV) return k + _V.GetOffset();
        IF_PARAM_ON_CONDITION(!_fixedScale) return _s.GetOffset();

        if (!_fixedXg)
        {
            if (_kg == FIXED_ROTATION) {}
            else IF_PARAM_ON_CONDITION(_kg == INDEPENDENT_ROTATION) return _Xg->GetOffset();
            else
            {
                // `_kg` in {BASIS_ROTATION_1=1, BASIS_ROTATION_2=2, BASIS_ROTATION_3=3}
                for (int n=0; n < _kg; n++)
                {
                    IF_PARAM_ON_CONDITION(!_fixedXgb) return _Xgb[n]->GetOffset();
                    IF_PARAM_ON_CONDITION(true) return _yg[n]->GetOffset();
                }
            }
        }

        IF_PARAM_ON_CONDITION(!_fixedTranslation) return _dg.GetOffset();
        IF_PARAM_ON_CONDITION(!_fixedV1) return k + _V1.GetOffset();

        #undef IF_PARAM_ON_CONDITION

        assert(false);

        return -1;
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _mesh.GetNumberOfVertices();
    }

    virtual void GetGlobalRotation_Unsafe(double * xg, double * qg) const
    {
        fillVector_Static<double, 3>(0., xg);
        if (_kg == FIXED_ROTATION) {}
        else if (_kg == INDEPENDENT_ROTATION)
        {
            copyVector_Static<double, 3>(_Xg->GetRotation(0), xg);
        }
        else
        {
            _XgComp.Rotation_Unsafe(0, xg);
        }

        quat_Unsafe(xg, qg);
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        double xg[3], qg[4];
        GetGlobalRotation_Unsafe(xg, qg);

        linearDeformationResiduals_Unsafe(_V.GetVertex(k), _V1.GetVertex(k), _w, 
                                          qg, _s.GetScale(), _dg.GetVertex(0), &e[0]);

    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int __ARG_COUNT = 0;
        #define IF_PARAM_ON_CONDITION(X) if ((X) && whichParam == __ARG_COUNT++)

        double xg[3], qg[4];
        GetGlobalRotation_Unsafe(xg, qg);

        IF_PARAM_ON_CONDITION(!_fixedV)
            return linearDeformationJac_V_Unsafe(_w, qg, _s.GetScale(), J[0]);

        IF_PARAM_ON_CONDITION(!_fixedScale)
            return linearDeformationJac_s_Unsafe(_w, qg, _V.GetVertex(k), J[0]);

        if (!_fixedXg)
        {
            if (_kg == FIXED_ROTATION) {}
            else IF_PARAM_ON_CONDITION(_kg == INDEPENDENT_ROTATION) 
            {
                // dr/dqg
                double Jq[12];
                linearDeformationJac_Q_Unsafe(_V.GetVertex(k), _w * _s.GetScale(), qg, Jq);

                // dqg/xg
                double Jxg[12];
                quatDx_Unsafe(xg, Jxg);

                multiply_A_B_Static<double, 3, 4, 3>(Jq, Jxg, J[0]);
                return;
            }
            else
            {
                // `_kg` in {BASIS_ROTATION_1=1, BASIS_ROTATION_2=2, BASIS_ROTATION_3=3}
                for (int n=0; n < _kg; n++)
                {
                    IF_PARAM_ON_CONDITION(!_fixedXgb) 
                    {
                        // dr/dqg
                        double Jq[12];
                        linearDeformationJac_Q_Unsafe(_V.GetVertex(k), _w * _s.GetScale(), qg, Jq);

                        // dqg/xg
                        double Jxg[12];
                        quatDx_Unsafe(xg, Jxg);

                        // dxg/dxgb
                        double Jxgb[9];
                        _XgComp.Jacobian_Unsafe(n, 0, true, Jxgb);

                        double A[9];
                        multiply_A_B_Static<double, 4, 3, 3>(Jxg, Jxgb, A);
                        multiply_A_B_Static<double, 3, 4, 3>(Jq, A, J[0]);
                        return;
                    }

                    IF_PARAM_ON_CONDITION(true)
                    {
                        // dr/dqg
                        double Jq[12];
                        linearDeformationJac_Q_Unsafe(_V.GetVertex(k), _w * _s.GetScale(), qg, Jq);

                        // dqg/xg
                        double Jxg[12];
                        quatDx_Unsafe(xg, Jxg);

                        // dxg/dyg
                        double Jyg[3];
                        _XgComp.Jacobian_Unsafe(n, 0, false, Jyg);

                        double A[4];
                        multiply_A_B_Static<double, 4, 3, 1>(Jxg, Jyg, A);
                        multiply_A_B_Static<double, 3, 4, 1>(Jq, A, J[0]);
                        return;
                    }
                }
            }
        }

        IF_PARAM_ON_CONDITION(!_fixedTranslation)
        {
            fillVector_Static<double, 9>(0., J[0]);
            J[0][0] = - _w;
            J[1][1] = - _w;
            J[2][2] = - _w;
            return;
        }

        IF_PARAM_ON_CONDITION(!_fixedV1)
        {
            fillVector_Static<double, 9>(0., J[0]);
            J[0][0] = _w;
            J[1][1] = _w;
            J[2][2] = _w;
            return;
        }
        
        #undef IF_PARAM_ON_CONDITION
        assert(false);
    }

    enum
    { 
        INDEPENDENT_ROTATION=-1, 
        FIXED_ROTATION=0 
    };

protected:
    const VertexNode & _V; 
    const ScaleNode & _s; 
    const int _kg; 
    vector<const RotationNode *> _Xgb; 
    vector<const CoefficientsNode *> _yg; 
    const RotationNode * _Xg; 
    const VertexNode & _dg; 
    const VertexNode & _V1; 
    const Mesh & _mesh; 
    bool _fixedXgb; 
    bool _fixedXb; 
    bool _fixedV; 
    bool _fixedV1; 
    bool _fixedScale; 
    bool _fixedXg; 
    bool _fixedTranslation;

    const RotationComposition _XgComp;
};

#endif

