#ifndef __ARAP2_H__
#define __ARAP2_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/mesh.h"
#include "Geometry/quaternion.h"
#include "Geometry/axis_angle.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <iterator>
using namespace std;

// arapResiduals_Unsafe
inline void arapResiduals_Unsafe(const double * Vi, const double * Vj,
           const double * V1i, const double * V1j,
           const double w, const double * q, 
           const double s, double * e)
{
    // BEGIN SymPy
    double x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11;

    x0 = Vi[2] - Vj[2];
    x1 = Vi[1] - Vj[1];
    x2 = Vi[0] - Vj[0];
    x3 = q[2]*q[2];
    x4 = q[2]*q[3];
    x5 = q[0]*q[1];
    x6 = q[0]*q[2];
    x7 = q[1]*q[3];
    x8 = q[1]*q[2];
    x9 = q[0]*q[3];
    x10 = q[1]*q[1];
    x11 = q[0]*q[0];

    e[0] = w*(V1i[0] - V1j[0] - s*(x0*(2*x6 + 2*x7) + x1*(-2*x4 + 2*x5) + x2*(-2*x10 - 2*x3 + 1)));
    e[1] = w*(V1i[1] - V1j[1] - s*(x0*(2*x8 - 2*x9) + x1*(-2*x11 - 2*x3 + 1) + x2*(2*x4 + 2*x5)));
    e[2] = w*(V1i[2] - V1j[2] - s*(x0*(-2*x10 - 2*x11 + 1) + x1*(2*x8 + 2*x9) + x2*(2*x6 - 2*x7)));
    // END SymPy
}

// arapJac_V_Unsafe
inline void arapJac_V_Unsafe(bool isVi, const double w, const double * q, const double s, double * J)
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

    double w_ = isVi ? -w : w;
    scaleVectorIP_Static<double, 9>(s * w_, J);
}

// arapJac_s_Unsafe
inline void arapJac_s_Unsafe(const double w, const double * q,    
                             const double * Vi, const double * Vj,
                             double * J)
{
    double R[9];
    rotationMatrix_Unsafe(q, R);

    double Vij[3];
    subtractVectors_Static<double, 3>(Vi, Vj, Vij);

    multiply_A_v_Static<double, 3, 3>(R, Vij, J);
    scaleVectorIP_Static<double, 3>(-w, J);
}

// arapJac_Q_Unsafe
inline void arapJac_Q_Unsafe(const double * Vi, const double * Vj,
                 const double w, const double * q, 
                 double * Jq)
{
    double x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14;

    // BEGIN SymPy
    x0 = Vi[2] - Vj[2];
    x1 = Vi[1] - Vj[1];
    x2 = Vi[0] - Vj[0];
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

// arapJac_X_Unsafe
inline void arapJac_X_Unsafe(const double * Vi, const double * Vj,
                 const double w, const double * q, const double * D, 
                 double * J)
{
    double Jq[12];

    arapJac_Q_Unsafe(Vi, Vj, w, q, Jq);
    
    for (int i=0; i<3; i++)
    {
        double * Ji = J + 3*i;
        const double * Jqi = Jq + 4*i;

        Ji[0] = Jqi[0]*D[0] + Jqi[1]*D[3] + Jqi[2]*D[6] + Jqi[3]*D[9];
        Ji[1] = Jqi[0]*D[1] + Jqi[1]*D[4] + Jqi[2]*D[7] + Jqi[3]*D[10];
        Ji[2] = Jqi[0]*D[2] + Jqi[1]*D[5] + Jqi[2]*D[8] + Jqi[3]*D[11];
    }
}

// arapJac_V1_Unsafe
inline void arapJac_V1_Unsafe(bool isV1i, const double w, double * J)
{
    double w_ = isV1i ? w : -w;

    for (int i=0; i<3; i++)
    {
        double * Ji = 3*i + J;

        Ji[0] = 0;
        Ji[1] = 0;
        Ji[2] = 0;

        Ji[i] = w_;
    }
}

// CompleteSectionedArapEnergy
class CompleteSectionedArapEnergy : public Energy
{
public:
    CompleteSectionedArapEnergy(const VertexNode & V, 
                                const ScaleNode & s,
                                const int kg, vector<const RotationNode *> && Xgb, vector<const CoefficientsNode *> && yg, const RotationNode * Xg,
                                const Vector<int> & k, const RotationNode & Xb, const CoefficientsNode & y, const RotationNode & X,
                                const VertexNode & V1,
                                const Mesh & mesh, const double w,
                                bool uniformWeights, 
                                bool fixedXgb, bool fixedXb, bool fixedV, bool fixedV1, bool fixedScale, bool fixedXg)
        : _V(V), _s(s), _kg(kg), _Xgb(Xgb), _yg(yg), _Xg(Xg), 
          _k(k), _Xb(Xb), _y(y), _X(X), 
          _V1(V1), 
          _mesh(mesh), _w(w), 
          _uniformWeights(uniformWeights), 
          _fixedXgb(fixedXgb), _fixedXb(fixedXb), _fixedV(fixedV), _fixedV1(fixedV1), _fixedScale(fixedScale), _fixedXg(fixedXg),
          _kLookup(V.GetCount())
    {
        int j = 0;
        for (int i = 0; i < _V.GetCount(); i++)
        {
            _kLookup[i] = j;

            if (_k[j] == FIXED_ROTATION)
                j += 1;
            else if (_k[j] == INDEPENDENT_ROTATION)
                j += 2;
            else
                j += 1 + 2 * _k[j];
        }
    }

    vector<int> * Initialise_UsedParamTypes() const
    {
        vector<int> * pUsedParamTypes = new vector<int>;

        if (!_fixedV)
        {
            pUsedParamTypes->push_back(_V.GetParamId());
            pUsedParamTypes->push_back(_V.GetParamId());
        }

        if (!_fixedScale) pUsedParamTypes->push_back(_s.GetParamId());

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

        return pUsedParamTypes;
    }

    void Finalise_UsedParamTypes(vector<int> * pUsedParamTypes) const
    {
        if (!_fixedV1)
        {
            pUsedParamTypes->push_back(_V1.GetParamId());
            pUsedParamTypes->push_back(_V1.GetParamId());
        }
    }

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * residualMaps[NUM_ROTATION_TYPES] = {nullptr};

        // there is a residual for each half-edge
        for (int r=0; r < _mesh.GetNumberOfHalfEdges(); r++)
        {
            // get the rotation for the given half edge
            int i = _mesh.GetHalfEdge(r, 0);

            // get the index into the residual maps for the given vertex
            int j = _k[_kLookup[i]] + 1;

            // assign this residual index to the residual map for the rotation type
            if (residualMaps[j] == nullptr)
                residualMaps[j] = new vector<int>;

            residualMaps[j]->push_back(r);
        }

        // construct the cost functions
        if (residualMaps[FIXED_ROTATION + 1] != nullptr && residualMaps[FIXED_ROTATION + 1]->size() > 0)
        {
            auto pUsedParamTypes = Initialise_UsedParamTypes();
            Finalise_UsedParamTypes(pUsedParamTypes);
            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, residualMaps[FIXED_ROTATION + 1]));
        }

        if (residualMaps[INDEPENDENT_ROTATION + 1] != nullptr && residualMaps[INDEPENDENT_ROTATION + 1]->size() > 0)
        {
            auto pUsedParamTypes = Initialise_UsedParamTypes();
            pUsedParamTypes->push_back(_X.GetParamId());
            Finalise_UsedParamTypes(pUsedParamTypes);
            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, residualMaps[INDEPENDENT_ROTATION + 1]));
        }

        for (int n=BASIS_ROTATION_1; n <= BASIS_ROTATION_3; n++)
        {
            if (residualMaps[n + 1] != nullptr && residualMaps[n + 1]->size() > 0)
            {
                auto pUsedParamTypes = Initialise_UsedParamTypes();

                for (int j=0; j < n; j++)
                {
                    if (!_fixedXb) pUsedParamTypes->push_back(_Xb.GetParamId());
                    pUsedParamTypes->push_back(_y.GetParamId());
                }

                Finalise_UsedParamTypes(pUsedParamTypes);
                costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, residualMaps[n + 1]));
            }
        }
    }

    virtual int GetCorrespondingParam(const int k, const int l) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);

        int __ARG_COUNT = 0;
        #define IF_PARAM_ON_CONDITION(X) if ((X) && l == __ARG_COUNT++)

        IF_PARAM_ON_CONDITION(!_fixedV) return i + _V.GetOffset();
        IF_PARAM_ON_CONDITION(!_fixedV) return j + _V.GetOffset();
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

        // `m` is the index into `k` for vertex `i`
        int m = _kLookup[i];
        if (_k[m] == FIXED_ROTATION) {}
        else IF_PARAM_ON_CONDITION(_k[m] == INDEPENDENT_ROTATION) return _k[m + 1] + _X.GetOffset();
        else
        {
            // `_k[m]` in {BASIS_ROTATION_1=1, BASIS_ROTATION_2=2, BASIS_ROTATION_3=3}
            for (int n=0; n < _k[m]; n++)
            {
                IF_PARAM_ON_CONDITION(!_fixedXb) return _k[m + 1 + 2*n] + _Xb.GetOffset();
                IF_PARAM_ON_CONDITION(true) return _k[m + 1 + 2*n + 1] + _y.GetOffset();
            }
        }

        IF_PARAM_ON_CONDITION(!_fixedV1) return i + _V1.GetOffset();
        IF_PARAM_ON_CONDITION(!_fixedV1) return j + _V1.GetOffset();
        #undef IF_PARAM_ON_CONDITION

        assert(false);

        return -1;
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _mesh.GetNumberOfHalfEdges();
    }

    virtual double GetEdgeWeight(int k) const
    {
        double w = _w;
        if (!_uniformWeights)
            w *= sqrt(_mesh.GetCotanWeight(_V.GetVertices(), k));

        return w;
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
            for (int n=0; n < _kg; n++)
                makeInterpolatedVector_Static<double, 3>(1.0, xg, 
                                                         _yg[n]->GetCoefficient(0), 
                                                         _Xgb[n]->GetRotation(0), 
                                                         xg);
        }

        quat_Unsafe(xg, qg);
    }

    virtual void GetRotationAtVertex_Unsafe(int i, double * xi, double * qi) const
    {
        fillVector_Static<double, 3>(0., xi);

        int m = _kLookup[i];
        if (_k[m] == FIXED_ROTATION) {}
        else if (_k[m] == INDEPENDENT_ROTATION)
        {
            copyVector_Static<double, 3>(_X.GetRotation(_k[m + 1]), xi);
        }
        else
        {
            for (int n=0; n < _k[m]; n++)
                makeInterpolatedVector_Static<double, 3>(1.0, xi, 
                                                         _y.GetCoefficient(_k[m + 1 + 2*n + 1]), 
                                                         _Xb.GetRotation(_k[m + 1 + 2*n]), 
                                                         xi);
        }

        quat_Unsafe(xi, qi);
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);

        double xg[3], qg[4];
        GetGlobalRotation_Unsafe(xg, qg);

        double xi[3], qi[4];
        GetRotationAtVertex_Unsafe(i, xi, qi);

        double q[4];
        quatMultiply_Unsafe(qg, qi, q);

        arapResiduals_Unsafe(_V.GetVertex(i), _V.GetVertex(j),
                             _V1.GetVertex(i), _V1.GetVertex(j),
                             GetEdgeWeight(k), q, _s.GetScale(), &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = GetEdgeWeight(k);

        int __ARG_COUNT = 0;
        #define IF_PARAM_ON_CONDITION(X) if ((X) && whichParam == __ARG_COUNT++)

        double xg[3], qg[4];
        GetGlobalRotation_Unsafe(xg, qg);

        double xi[3], qi[4];
        GetRotationAtVertex_Unsafe(i, xi, qi);

        double q[4];
        quatMultiply_Unsafe(qg, qi, q);

        // Vi
        IF_PARAM_ON_CONDITION(!_fixedV) return arapJac_V_Unsafe(true, w, q, _s.GetScale(), J[0]);

        // Vj
        IF_PARAM_ON_CONDITION(!_fixedV) return arapJac_V_Unsafe(false, w, q, _s.GetScale(), J[0]);

        // s
        IF_PARAM_ON_CONDITION(!_fixedScale) return arapJac_s_Unsafe(w, q, _V.GetVertex(i), _V.GetVertex(j), J[0]);

        if (!_fixedXg)
        {
            if (_kg == FIXED_ROTATION) {}
            else IF_PARAM_ON_CONDITION(_kg == INDEPENDENT_ROTATION) 
            {
                // Xg

                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                // dq/dqg
                double Jqg[16];
                quatMultiply_dp_Unsafe(qi, Jqg);
                
                // dqg/xg [xg == _Xg->GetRotation(0)]
                double Jxg[12];
                quatDx_Unsafe(_Xg->GetRotation(0), Jxg);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Jqg, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Jxg, J[0]);
                return;
            }
            else
            {
                for (int n=0; n < _kg; n++)
                {
                    IF_PARAM_ON_CONDITION(!_fixedXgb)
                    {
                        // Xgb

                        // dr/dq
                        double Jq[12];
                        arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                        // dq/dqg
                        double Jqg[16];
                        quatMultiply_dp_Unsafe(qi, Jqg);

                        // dqg/dxg 
                        double Jxg[12];
                        quatDx_Unsafe(xg, Jxg);

                        // dxg/dxgb == _yg[n]->GetCoefficient(0)
                        scaleVectorIP_Static<double, 12>(_yg[n]->GetCoefficient(0), Jxg);

                        double A[12];
                        multiply_A_B_Static<double, 3, 4, 4>(Jq, Jqg, A);
                        multiply_A_B_Static<double, 3, 4, 3>(A, Jxg, J[0]);
                        return;
                    }

                    IF_PARAM_ON_CONDITION(true)
                    {
                        // yg

                        // dr/dq
                        double Jq[12];
                        arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                        // dq/dqg
                        double Jqg[16];
                        quatMultiply_dp_Unsafe(qi, Jqg);

                        // dqg/dxg 
                        double Jxg[12];
                        quatDx_Unsafe(xg, Jxg);

                        // dxg/dyg == _Xgb[n]->GetRotation(0)

                        double A[12], B[9];
                        multiply_A_B_Static<double, 3, 4, 4>(Jq, Jqg, A);
                        multiply_A_B_Static<double, 3, 4, 3>(A, Jxg, B);
                        multiply_A_v_Static<double, 3, 3>(B, _Xgb[n]->GetRotation(0), J[0]);
                        return;
                    }
                }
            }
        }

        int m = _kLookup[i];
        if (_k[m] == FIXED_ROTATION) {}
        else IF_PARAM_ON_CONDITION(_k[m] == INDEPENDENT_ROTATION)
        {
            // X

            // dr/dq
            double Jq[12];
            arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

            // dq/dqi
            double Jqi[16];
            quatMultiply_dq_Unsafe(qg, Jqi);

            // dqi/xi [xi == _X.GetRotation(_k[m + 1])]
            double Jxi[12];
            quatDx_Unsafe(_X.GetRotation(_k[m + 1]), Jxi);

            double A[12];
            multiply_A_B_Static<double, 3, 4, 4>(Jq, Jqi, A);
            multiply_A_B_Static<double, 3, 4, 3>(A, Jxi, J[0]);
            return;
        }
        else
        {
            for (int n=0; n < _k[m]; n++)
            {
                IF_PARAM_ON_CONDITION(!_fixedXgb)
                {
                    // Xb

                    // dr/dq
                    double Jq[12];
                    arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                    // dq/dqi
                    double Jqi[16];
                    quatMultiply_dq_Unsafe(qg, Jqi);

                    // dqi/xi 
                    double Jxi[12];
                    quatDx_Unsafe(xi, Jxi);

                    // dxi/dxb == _y.GetCoefficient(_k[m + 1 + 2*n + 1])
                    scaleVectorIP_Static<double, 12>(_y.GetCoefficient(_k[m + 1 + 2*n + 1]), Jxi);

                    double A[12];
                    multiply_A_B_Static<double, 3, 4, 4>(Jq, Jqi, A);
                    multiply_A_B_Static<double, 3, 4, 3>(A, Jxi, J[0]);
                    return;
                }

                IF_PARAM_ON_CONDITION(true)
                {
                    // y

                    // dr/dq
                    double Jq[12];
                    arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                    // dq/dqi
                    double Jqi[16];
                    quatMultiply_dq_Unsafe(qg, Jqi);

                    // dqi/xi 
                    double Jxi[12];
                    quatDx_Unsafe(xi, Jxi);

                    double A[12], B[9];
                    multiply_A_B_Static<double, 3, 4, 4>(Jq, Jqi, A);
                    multiply_A_B_Static<double, 3, 4, 3>(A, Jxi, B);
                    multiply_A_v_Static<double, 3, 3>(B, _Xb.GetRotation(_k[m + 1 + 2*n]), J[0]);
                    return;
                }
            }
        }

        IF_PARAM_ON_CONDITION(!_fixedV1) return arapJac_V1_Unsafe(true, w, J[0]);
        IF_PARAM_ON_CONDITION(!_fixedV1) return arapJac_V1_Unsafe(false, w, J[0]);

        #undef IF_PARAM_ON_CONDITION

        assert(false);
    }

    enum
    { 
        INDEPENDENT_ROTATION=-1, 
        FIXED_ROTATION=0, 
        BASIS_ROTATION_1=1, 
        BASIS_ROTATION_2=2, 
        BASIS_ROTATION_3=3, 
        NUM_ROTATION_TYPES=5 
    };

protected:
    const VertexNode & _V;
    const ScaleNode & _s;
    const int _kg;
    const vector<const RotationNode *> _Xgb;
    const vector<const CoefficientsNode *> _yg;
    const RotationNode * _Xg;
    const Vector<int> & _k;
    const RotationNode & _Xb;
    const CoefficientsNode & _y;
    const RotationNode & _X;
    const VertexNode & _V1;
    const Mesh & _mesh;
    const double _w;
    bool _uniformWeights;
    bool _fixedXgb;
    bool _fixedXb;
    bool _fixedV;
    bool _fixedV1;
    bool _fixedScale;
    bool _fixedXg;

    Vector<int> _kLookup;
};

// RigidTransformArapEnergy
class RigidTransformArapEnergy : public Energy
{
public:
    RigidTransformArapEnergy(const VertexNode & V, 
                             const ScaleNode & s,
                             const RotationNode & Xg,
                             const RotationNode & X,
                             const VertexNode & V1,
                             const Mesh & mesh, const double w,
                             bool uniformWeights,
                             bool fixedScale)

    : _V(V), _s(s), _Xg(Xg), _X(X), _V1(V1), _mesh(mesh), _w(w), _uniformWeights(uniformWeights), _fixedScale(fixedScale)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_Xg.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        if (!_fixedScale)
            pUsedParamTypes->push_back(_s.GetParamId());

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        switch (i)
        {
        case 0:
            return _mesh.GetHalfEdge(k, 0) + _X.GetOffset();
        case 1:
            return _Xg.GetOffset(); 
        case 2:
            return _mesh.GetHalfEdge(k, 0) + _V1.GetOffset();
        case 3:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
        case 4:
            return _s.GetOffset(); 
        }

        assert(false);

        return -1;
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _mesh.GetNumberOfHalfEdges();
    }

    virtual double GetEdgeWeight(int k) const
    {
        double w = _w;
        if (!_uniformWeights)
            w *= sqrt(_mesh.GetCotanWeight(_V.GetVertices(), k));

        return w;
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = GetEdgeWeight(k);

        double qi[4];
        quat_Unsafe(_X.GetRotation(i), qi);

        double qg[4];
        quat_Unsafe(_Xg.GetRotation(0), qg);

        double q[4];
        quatMultiply_Unsafe(qg, qi, q);

        arapResiduals_Unsafe(_V.GetVertex(i), _V.GetVertex(j),
                             _V1.GetVertex(i), _V1.GetVertex(j),
                             w, q, _s.GetScale(), &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = GetEdgeWeight(k);

        double qi[4];
        quat_Unsafe(_X.GetRotation(i), qi);

        double qg[4];
        quat_Unsafe(_Xg.GetRotation(0), qg);

        double q[4];
        quatMultiply_Unsafe(qg, qi, q);

        switch (whichParam)
        {
        case 0:
            // X
            {
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                // dq/dqi
                double Jqi[16];
                quatMultiply_dq_Unsafe(qg, Jqi);

                // dqi/xi
                double Jxi[12];
                quatDx_Unsafe(_X.GetRotation(i), Jxi);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Jqi, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Jxi, J[0]);

                return;
            }
        case 1:
            // Xg
            {
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                // dq/dqg
                double Jqg[16];
                quatMultiply_dp_Unsafe(qi, Jqg);

                // dqg/xg
                double Jxg[12];
                quatDx_Unsafe(_Xg.GetRotation(0), Jxg);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Jqg, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Jxg, J[0]);

                return;
            }

        case 2:
            // V1i
            {
                
                arapJac_V1_Unsafe(true, w, J[0]);
                return;
            }
        case 3:
            // V1j
            {
                arapJac_V1_Unsafe(false, w, J[0]);
                return;
            }

        case 4:
            // s
            {
                arapJac_s_Unsafe(w, q, _V.GetVertex(i), _V.GetVertex(j), J[0]);
                return;
            }
        }

        assert(false);
    }

protected:
    const VertexNode & _V;
    const ScaleNode & _s;
    const RotationNode & _Xg;
    const RotationNode & _X;
    const VertexNode & _V1;
    const Mesh & _mesh;
    const double _w;
    bool _uniformWeights;
    bool _fixedScale;
};

// ScaleRegulariseEnergy
class ScaleRegulariseEnergy : public Energy
{
public:
    ScaleRegulariseEnergy(const ScaleNode & s, const double w)
        : _s(s), _w(w)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>(1, _s.GetParamId());
        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 1));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        return _s.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        return 1;
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        e[0] = _w * log(_s.GetScale());
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        J[0][0] = _w * (1.0 / _s.GetScale());
    }

protected:
    const ScaleNode & _s;
    const double _w;
};

// RotationRegulariseEnergy
class RotationRegulariseEnergy : public Energy
{
public:
    RotationRegulariseEnergy(const RotationNode & X, const double w)
        : _X(X), _w(w)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>(1, _X.GetParamId());
        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        return k + _X.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _X.GetCount();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        const double * X = _X.GetRotation(k);

        for (int i=0; i < e.size(); ++i)
            e[i] = _w * X[i];
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        fillMatrix(J, 0.);
        for (int i=0; i < J.num_rows(); ++i)
            J[i][i] = _w;
    }

protected:
    const RotationNode & _X;
    const double _w;
};

// GlobalRotationLinearCombinationEnergy
class GlobalRotationLinearCombinationEnergy : public Energy
{
public:
    GlobalRotationLinearCombinationEnergy(vector<int> && kg, 
                                          vector<vector<const RotationNode *>> &&  Xgb, 
                                          vector<vector<const CoefficientsNode *>> && yg, 
                                          vector<const RotationNode *> && Xg,
                                          const Vector<double> && A,
                                          const double w,
                                          const Vector<int> && fixed,
                                          bool fixedXgb)
    : _kg(kg), _Xgb(Xgb), _yg(yg), _Xg(Xg), _A(A), _w(w), _fixed(fixed), _fixedXgb(fixedXgb)
    {
        for (int i=0; i < _fixed.size(); i++)
        {
            if (_fixed[i])
                continue;

            int n = 0;
            switch (_kg[i])
            {
            case FIXED_ROTATION:
                continue;
            case INDEPENDENT_ROTATION:
            {
                _paramMap.push_back(pair<int, int>(i, 0));
                continue;
            }
            default:
                n = _kg[i];
                break;
            }

            int l = 0;

            for (int j=0; j < n; j++)
            {
                if (!_fixedXgb) _paramMap.push_back(pair<int, int>(i, l++));
                _paramMap.push_back(pair<int, int>(i, l++));
            }
        }
    }

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        
        for (int i=0; i < _fixed.size(); i++)
        {
            if (_fixed[i])
                continue;

            int n = 0;
            switch (_kg[i])
            {
            case FIXED_ROTATION:
                continue;
            case INDEPENDENT_ROTATION:
            {
                pUsedParamTypes->push_back(_Xg[i]->GetParamId());
                continue;
            }
            default:
                n = _kg[i];
                break;
            }

            int l = 0;

            for (int j=0; j < n; j++)
            {
                if (!_fixedXgb) pUsedParamTypes->push_back(_Xgb[i][j]->GetParamId());

                pUsedParamTypes->push_back(_yg[i][j]->GetParamId());
            }
        }

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int l) const
    {
        auto entry = _paramMap[l];
        int i = entry.first;

        if (_kg[i] == FIXED_ROTATION)
            assert(false);
        else if (_kg[i] == INDEPENDENT_ROTATION)
            return _Xg[i]->GetOffset();

        int __ARG_COUNT = 0;
        #define IF_PARAM_ON_CONDITION(X) if ((X) && (entry.second == __ARG_COUNT++))

        for (int j=0; j < _kg[i]; j++)
        {
            IF_PARAM_ON_CONDITION(!_fixedXgb) return _Xgb[i][j]->GetOffset();
            IF_PARAM_ON_CONDITION(true) return _yg[i][j]->GetOffset();
        }

        #undef IF_PARAM_ON_CONDITION

        assert(false);

        return -1;
    }

    virtual int GetNumberOfMeasurements() const
    {
        return 1;
    }

    virtual void GetGlobalRotation_Unsafe(const int i, double * xgi) const
    {
        fillVector_Static<double, 3>(0., xgi);

        if (_kg[i] == FIXED_ROTATION) {}
        else if (_kg[i] == INDEPENDENT_ROTATION)
        {
            copyVector_Static<double, 3>(_Xg[i]->GetRotation(0), xgi);
        }
        else
        {
            for (int n=0; n < _kg[i]; n++)
                makeInterpolatedVector_Static<double, 3>(1.0, xgi, 
                                                         _yg[i][n]->GetCoefficient(0), 
                                                         _Xgb[i][n]->GetRotation(0), 
                                                         xgi);
        }
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        double xg[3] = {0., 0., 0.};

        for (int i = 0; i < _fixed.size(); i++)
        {
            double xgi[3];
            GetGlobalRotation_Unsafe(i, xgi);
            axScale_Unsafe(_A[i], xgi, xgi);
            axAdd_Unsafe(xgi, xg, xg);
        }

        scaleVectorIP_Static<double, 3>(_w, xg);
        copyVector_Static<double, 3>(xg, &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        auto entry = _paramMap[whichParam];
        int i = entry.first;

        if (_kg[i] == FIXED_ROTATION)
            assert(false);

        double xg[3] = {0.};
        for (int r = 0; r < i; r++)
        {
            double xgr[3];
            GetGlobalRotation_Unsafe(r, xgr);
            axScale_Unsafe(_A[r], xgr, xgr);
            axAdd_Unsafe(xgr, xg, xg);
        }

        double xgi[3];
        GetGlobalRotation_Unsafe(i, xgi);
        axScale_Unsafe(_A[i], xgi, xgi);

        double Jq[9] = {0.};
        axAdd_da_Unsafe(xgi, xg, Jq);
        scaleVectorIP_Static<double, 9>(_w * _A[i], Jq);

        axAdd_Unsafe(xgi, xg, xg);

        for (int l = i + 1; l < _fixed.size(); l++)
        {
            double xgl[3];
            GetGlobalRotation_Unsafe(l, xgl);
            axScale_Unsafe(_A[l], xgl, xgl);

            double Jp[9];
            axAdd_db_Unsafe(xgl, xg, Jp);

            double Jr[9];
            multiply_A_B_Static<double, 3, 3, 3>(Jp, Jq, Jr);
            copyVector_Static<double, 9>(Jr, Jq);

            axAdd_Unsafe(xgl, xg, xg);
        }

        if (_kg[i] == INDEPENDENT_ROTATION)
        {
            copyVector_Static<double, 9>(Jq, J[0]);
            return;
        }
        
        int __ARG_COUNT = 0;
        #define IF_PARAM_ON_CONDITION(X) if ((X) && (entry.second == __ARG_COUNT++))

        for (int j=0; j < _kg[i]; j++)
        {
            IF_PARAM_ON_CONDITION(!_fixedXgb)
            {
                scaleVectorIP_Static<double, 9>(_yg[i][j]->GetCoefficient(0), Jq);
                copyVector_Static<double, 9>(Jq, J[0]);
                return;
            }

            IF_PARAM_ON_CONDITION(true) 
            {
                multiply_A_v_Static<double, 3, 3>(Jq, _Xgb[i][j]->GetRotation(0), J[0]);
                return; 
            }
        }

        #undef IF_PARAM_ON_CONDITION
    }

    enum
    { 
        INDEPENDENT_ROTATION=-1, 
        FIXED_ROTATION=0, 
        BASIS_ROTATION_1=1, 
        BASIS_ROTATION_2=2, 
        BASIS_ROTATION_3=3, 
        NUM_ROTATION_TYPES=5 
    };

protected:
    vector<int> _kg;
    vector<vector<const RotationNode *>> _Xgb;
    vector<vector<const CoefficientsNode *>> _yg;
    vector<const RotationNode *> _Xg;
    const Vector<double> _A;
    const double _w;
    const Vector<int> _fixed;
    bool _fixedXgb;

    vector<pair<int, int>> _paramMap;
};

// GlobalScalesLinearCombinationEnergy
class GlobalScalesLinearCombinationEnergy : public Energy
{
public:
    GlobalScalesLinearCombinationEnergy(vector<const ScaleNode *> && s,
                                        const Vector<double> && A,
                                        const double w,
                                        const Vector<int> && fixed)

    : _s(s), _A(A), _w(w), _fixed(fixed) 
    {
        for (int i=0; i < _fixed.size(); i++)
        {
            if (_fixed[i])
                continue;

            _paramMap.push_back(i);
        }
    }

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;

        for (int i=0; i < _fixed.size(); i++)
        {
            if (_fixed[i])
                continue;

            pUsedParamTypes->push_back(_s[i]->GetParamId());
        }

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 1));
    }

    virtual int GetCorrespondingParam(const int k, const int l) const
    {
        int i = _paramMap[l];
        return _s[i]->GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        return 1;
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        double r = 0.;
        for (int i = 0; i < _A.size(); i++)
            r += _A[i] * _s[i]->GetScale();

        r *= _w;

        e[0] = r;
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int i = _paramMap[whichParam];
        J[0][0] = _w * _A[i];
    }

protected:
    vector<const ScaleNode *> _s;
    const Vector<double> _A;
    const double _w;
    const Vector<int> _fixed;

    vector<int> _paramMap;
};

#endif
