#ifndef __ARAP_H__
#define __ARAP_H__

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

// arapResiduals_ScaleV_Unsafe
inline void arapResiduals_ScaleV_Unsafe(const double * Vi, const double * Vj,
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

    e[0] = w*(s*(V1i[0] - V1j[0]) - (x0*(2*x6 + 2*x7) + x1*(-2*x4 + 2*x5) + x2*(-2*x10 - 2*x3 + 1)));
    e[1] = w*(s*(V1i[1] - V1j[1]) - (x0*(2*x8 - 2*x9) + x1*(-2*x11 - 2*x3 + 1) + x2*(2*x4 + 2*x5)));
    e[2] = w*(s*(V1i[2] - V1j[2]) - (x0*(-2*x10 - 2*x11 + 1) + x1*(2*x8 + 2*x9) + x2*(2*x6 - 2*x7)));
    // END SymPy
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

// ARAPEnergy
class ARAPEnergy : public Energy
{
public:
    ARAPEnergy(const VertexNode & V,  const RotationNode & X, const VertexNode & V1,
               const Mesh & mesh, const double w)
        : _V(V), _X(X), _V1(V1), _mesh(mesh), _w(w)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        switch (i)
        {
        case 0:
            return _mesh.GetHalfEdge(k, 0) + _X.GetOffset();
        case 1:
            return _mesh.GetHalfEdge(k, 0) + _V1.GetOffset();
        case 2:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
        }

        assert(false);

        return -1;
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _mesh.GetNumberOfHalfEdges();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = _w * sqrt(_mesh.GetCotanWeight(_V.GetVertices(), k));

        double q[4];
        quat_Unsafe(_X.GetRotation(i), q);

        arapResiduals_Unsafe(_V.GetVertex(i), _V.GetVertex(j),
                             _V1.GetVertex(i), _V1.GetVertex(j),
                             w, q, 1.0, &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = _w * sqrt(_mesh.GetCotanWeight(_V.GetVertices(), k));

        double q[4];
        quat_Unsafe(_X.GetRotation(i), q);
    
        double D[12];
        quatDx_Unsafe(_X.GetRotation(i), D);

        switch (whichParam)
        {
        case 0:
            {
                arapJac_X_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w, q, D, J[0]);
                return;
            }
        case 1:
            {
                arapJac_V1_Unsafe(true, w, J[0]);
                return;
            }
        case 2:
            {
                arapJac_V1_Unsafe(false, w, J[0]);
                return;
            }
        }

        assert(false);
    }

protected:
    const VertexNode & _V;
    const RotationNode & _X;
    const VertexNode & _V1;

    const Mesh & _mesh;
    const double _w;
};

// RigidTransformARAPEnergy
class RigidTransformARAPEnergy: public Energy
{
public:
    RigidTransformARAPEnergy(const VertexNode & V,  const RotationNode & X, 
                             const RotationNode & Xg, const ScaleNode & s,
                             const VertexNode & V1, const Mesh & mesh, const double w,
                             bool uniformWeights)
        : _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), _w(w), _uniformWeights(uniformWeights)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_Xg.GetParamId());
        pUsedParamTypes->push_back(_s.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());

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
            return _s.GetOffset(); 
        case 3:
            return _mesh.GetHalfEdge(k, 0) + _V1.GetOffset();
        case 4:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
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

        // qg * qi -> q
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
                double Dqi[16];
                quatMultiply_dq_Unsafe(qg, Dqi);

                // dqi/xi
                double Di[12];
                quatDx_Unsafe(_X.GetRotation(i), Di);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqi, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Di, J[0]);

                return;
            }
        case 1:
            // Xg
            {
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                // dq/dqg
                double Dqg[16];
                quatMultiply_dp_Unsafe(qi, Dqg);

                // dqg/xg
                double Dg[12];
                quatDx_Unsafe(_Xg.GetRotation(0), Dg);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqg, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Dg, J[0]);

                return;
            }
        case 2:
            // s
            {
                arapJac_s_Unsafe(w, q, _V.GetVertex(i), _V.GetVertex(j), J[0]);
                return;
            }

        case 3:
            // V1i
            {
                
                arapJac_V1_Unsafe(true, w, J[0]);
                return;
            }
        case 4:
            // V1j
            {
                arapJac_V1_Unsafe(false, w, J[0]);
                return;
            }
        }

        assert(false);
    }

protected:
    const VertexNode & _V;
    const RotationNode & _X;
    const RotationNode & _Xg;
    const ScaleNode & _s;
    const VertexNode & _V1;

    const Mesh & _mesh;
    const double _w;
    bool _uniformWeights;
};

// RigidTransformARAPEnergy2 (uses `arapResiduals_ScaleV_Unsafe` and `GlobalRotationNode`)
class RigidTransformARAPEnergy2: public Energy
{
public:
    RigidTransformARAPEnergy2(const VertexNode & V, const GlobalRotationNode & Xg, const ScaleNode & s,
                              const RotationNode & X, const VertexNode & V1, const Mesh & mesh, const double w,
                              bool uniformWeights)
        : _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), _w(w), _uniformWeights(uniformWeights)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_Xg.GetParamId());
        pUsedParamTypes->push_back(_s.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());

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
            return _s.GetOffset(); 
        case 3:
            return _mesh.GetHalfEdge(k, 0) + _V1.GetOffset();
        case 4:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
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
        quat_Unsafe(_Xg.GetRotation(), qg);

        double q[4];
        quatMultiply_Unsafe(qg, qi, q);

        arapResiduals_ScaleV_Unsafe(_V.GetVertex(i), _V.GetVertex(j),
                                    _V1.GetVertex(i), _V1.GetVertex(j),
                                    w, q, _s.GetScale(), &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = GetEdgeWeight(k);

        // qg * qi -> q
        double qi[4];
        quat_Unsafe(_X.GetRotation(i), qi);

        double qg[4];
        quat_Unsafe(_Xg.GetRotation(), qg);

        double q[4];
        quatMultiply_Unsafe(qg, qi, q);

        switch (whichParam)
        {
        case 0:
            // X
            {
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w, q, Jq);

                // dq/dqi
                double Dqi[16];
                quatMultiply_dq_Unsafe(qg, Dqi);

                // dqi/xi
                double Di[12];
                quatDx_Unsafe(_X.GetRotation(i), Di);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqi, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Di, J[0]);

                return;
            }
        case 1:
            // Xg
            {
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w, q, Jq);

                // dq/dqg
                double Dqg[16];
                quatMultiply_dp_Unsafe(qi, Dqg);

                // dqg/xg
                double Dg[12];
                quatDx_Unsafe(_Xg.GetRotation(), Dg);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqg, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Dg, J[0]);

                return;
            }
        case 2:
            // s
            {
                subtractVectors_Static<double, 3>(_V1.GetVertex(i), _V1.GetVertex(j), J[0]);
                scaleVectorIP_Static<double, 3>(w, J[0]);
                return;
            }

        case 3:
            // V1i
            {
                
                arapJac_V1_Unsafe(true, w * _s.GetScale(), J[0]);
                return;
            }
        case 4:
            // V1j
            {
                arapJac_V1_Unsafe(false, w * _s.GetScale(), J[0]);
                return;
            }
        }

        assert(false);
    }

protected:
    const VertexNode & _V;
    const RotationNode & _X;
    const GlobalRotationNode & _Xg;
    const ScaleNode & _s;
    const VertexNode & _V1;

    const Mesh & _mesh;
    const double _w;
    bool _uniformWeights;
};

// TODO DualArapEnergy::EvaluateJacobian DOES NOT take into account the cotangent weight
// dependence

// DualARAPEnergy
class DualARAPEnergy : public Energy
{
public:
    DualARAPEnergy(const VertexNode & V,  const RotationNode & X, const VertexNode & V1,
               const Mesh & mesh, const double w, bool uniformWeights = true)
        : _V(V), _X(X), _V1(V1), _mesh(mesh), _w(w), _uniformWeights(uniformWeights) 
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        switch (i)
        {
        case 0:
            return _mesh.GetHalfEdge(k, 0) + _X.GetOffset();
        case 1:
            return _mesh.GetHalfEdge(k, 0) + _V1.GetOffset();
        case 2:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
        case 3:
            return _mesh.GetHalfEdge(k, 0) + _V.GetOffset();
        case 4:
            return _mesh.GetHalfEdge(k, 1) + _V.GetOffset();
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

        double q[4];
        quat_Unsafe(_X.GetRotation(i), q);

        arapResiduals_Unsafe(_V.GetVertex(i), _V.GetVertex(j),
                             _V1.GetVertex(i), _V1.GetVertex(j),
                             w, q, 1.0, &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = GetEdgeWeight(k);

        double q[4];
        quat_Unsafe(_X.GetRotation(i), q);
    
        double D[12];
        quatDx_Unsafe(_X.GetRotation(i), D);

        switch (whichParam)
        {
        case 0:
            {
                arapJac_X_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w, q, D, J[0]);
                return;
            }
        case 1:
            {
                arapJac_V1_Unsafe(true, w, J[0]);
                return;
            }
        case 2:
            {
                arapJac_V1_Unsafe(false, w, J[0]);
                return;
            }
        case 3:
            {
                arapJac_V_Unsafe(true, w, q, 1.0, J[0]);
                return;
            }
        case 4:
            {
                arapJac_V_Unsafe(false, w, q, 1.0, J[0]);
                return;
            }
        }

        assert(false);
    }

protected:
    const VertexNode & _V;
    const RotationNode & _X;
    const VertexNode & _V1;

    const Mesh & _mesh;
    const double _w;

    bool _uniformWeights;
};

// DualScaledARAPEnergy
class DualScaledARAPEnergy : public Energy
{
public:
    DualScaledARAPEnergy(const VertexNode & V,  const RotationNode & X, const ScaleNode & s,
                         const VertexNode & V1, const Mesh & mesh, const double w, 
                         bool uniformWeights = true)
        : _V(V), _X(X), _s(s), _V1(V1), _mesh(mesh), _w(w), _uniformWeights(uniformWeights) 
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_s.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        switch (i)
        {
        case 0:
            return _mesh.GetHalfEdge(k, 0) + _X.GetOffset();
        case 1:
            return _s.GetOffset();
        case 2:
            return _mesh.GetHalfEdge(k, 0) + _V1.GetOffset();
        case 3:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
        case 4:
            return _mesh.GetHalfEdge(k, 0) + _V.GetOffset();
        case 5:
            return _mesh.GetHalfEdge(k, 1) + _V.GetOffset();
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

        double q[4];
        quat_Unsafe(_X.GetRotation(i), q);

        arapResiduals_Unsafe(_V.GetVertex(i), _V.GetVertex(j),
                             _V1.GetVertex(i), _V1.GetVertex(j),
                             w, q, _s.GetScale(), &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = GetEdgeWeight(k);

        double q[4];
        quat_Unsafe(_X.GetRotation(i), q);
    
        double D[12];
        quatDx_Unsafe(_X.GetRotation(i), D);

        switch (whichParam)
        {
        case 0:
            {
                arapJac_X_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, D, J[0]);
                return;
            }
        case 1:
            {
                arapJac_s_Unsafe(w, q, _V.GetVertex(i), _V.GetVertex(j), J[0]);
                return;
            }
        case 2:
            {
                arapJac_V1_Unsafe(true, w, J[0]);
                return;
            }
        case 3:
            {
                arapJac_V1_Unsafe(false, w, J[0]);
                return;
            }
        case 4:
            {
                arapJac_V_Unsafe(true, w, q, _s.GetScale(), J[0]);
                return;
            }
        case 5:
            {
                arapJac_V_Unsafe(false, w, q, _s.GetScale(), J[0]);
                return;
            }
        }

        assert(false);
    }

protected:
    const VertexNode & _V;
    const RotationNode & _X;
    const ScaleNode & _s;
    const VertexNode & _V1;

    const Mesh & _mesh;
    const double _w;

    bool _uniformWeights;
};

// DualRigidTransformArapEnergy
class DualRigidTransformArapEnergy: public Energy
{
public:
    DualRigidTransformArapEnergy(const VertexNode & V,  const RotationNode & X, 
                                 const RotationNode & Xg, const ScaleNode & s,
                                 const VertexNode & V1, const Mesh & mesh, const double w,
                                 bool uniformWeights = true)
        : _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), _w(w), _uniformWeights(uniformWeights)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_Xg.GetParamId());
        pUsedParamTypes->push_back(_s.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());

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
            return _s.GetOffset(); 
        case 3:
            return _mesh.GetHalfEdge(k, 0) + _V1.GetOffset();
        case 4:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
        case 5:
            return _mesh.GetHalfEdge(k, 0) + _V.GetOffset();
        case 6:
            return _mesh.GetHalfEdge(k, 1) + _V.GetOffset();
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

        arapResiduals_ScaleV_Unsafe(_V.GetVertex(i), _V.GetVertex(j),
                                    _V1.GetVertex(i), _V1.GetVertex(j),
                                    w, q, _s.GetScale(), &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = GetEdgeWeight(k);

        // qg * qi -> q
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
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w, q, Jq);

                // dq/dqi
                double Dqi[16];
                quatMultiply_dq_Unsafe(qg, Dqi);

                // dqi/xi
                double Di[12];
                quatDx_Unsafe(_X.GetRotation(i), Di);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqi, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Di, J[0]);

                return;
            }
        case 1:
            // Xg
            {
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w, q, Jq);

                // dq/dqg
                double Dqg[16];
                quatMultiply_dp_Unsafe(qi, Dqg);

                // dqg/xg
                double Dg[12];
                quatDx_Unsafe(_Xg.GetRotation(0), Dg);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqg, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Dg, J[0]);

                return;
            }
        case 2:
            // s
            {
                // arapJac_s_Unsafe(w, q, _V.GetVertex(i), _V.GetVertex(j), J[0]);
                subtractVectors_Static<double, 3>(_V1.GetVertex(i), _V1.GetVertex(j), J[0]);
                scaleVectorIP_Static<double, 3>(w, J[0]);
                return;
            }

        case 3:
            // V1i
            {
                
                arapJac_V1_Unsafe(true, w * _s.GetScale(), J[0]);
                return;
            }
        case 4:
            // V1j
            {
                arapJac_V1_Unsafe(false, w * _s.GetScale(), J[0]);
                return;
            }
        case 5:
            // Vi
            {
                arapJac_V_Unsafe(true, w, q, 1.0, J[0]);
                return;
            }
        case 6:
            // Vj
            {
                arapJac_V_Unsafe(false, w, q, 1.0, J[0]);
                return;
            }
        }

        assert(false);
    }

protected:
    const VertexNode & _V;
    const RotationNode & _X;
    const RotationNode & _Xg;
    const ScaleNode & _s;
    const VertexNode & _V1;

    const Mesh & _mesh;
    const double _w;
    bool _uniformWeights;
};

// DualNonLinearBasisArapEnergy
class DualNonLinearBasisArapEnergy : public Energy
{
public:
    DualNonLinearBasisArapEnergy(const VertexNode & V, const GlobalRotationNode & Xg, const ScaleNode & s,
                                 const vector<RotationNode *> & X, const CoefficientsNode & y,
                                 const VertexNode & V1, const Mesh & mesh, const double w,
                                 bool uniformWeights = true)
        : _V(V), _Xg(Xg), _s(s), _X(X), _y(y), _V1(V1), _mesh(mesh), 
          _w(w), _uniformWeights(uniformWeights), _n(X.size())
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_Xg.GetParamId());
        pUsedParamTypes->push_back(_s.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());
        pUsedParamTypes->push_back(_V1.GetParamId());

        for (int i=0; i < _n; ++i)
            pUsedParamTypes->push_back(_X[i]->GetParamId());

        for (int i=0; i < _n; ++i)
            pUsedParamTypes->push_back(_y.GetParamId());

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        switch (i)
        {
        case 0:
            return _Xg.GetOffset();
        case 1:
            return _s.GetOffset();
        case 2:
            return _mesh.GetHalfEdge(k, 0) + _V.GetOffset();
        case 3:
            return _mesh.GetHalfEdge(k, 1) + _V.GetOffset();
        case 4:
            return _mesh.GetHalfEdge(k, 0) + _V1.GetOffset();
        case 5:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
        default:
            break;
        }

        int j = i - 6;

        if (j < _n)
            return _mesh.GetHalfEdge(k, 0) + _X[j]->GetOffset();
        else 
            return _y.GetOffset() + (j - _n);
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

        double x[3] = {0.}, xl[3];

        for (int l = 0; l < _n; ++l)
        {
            axScale_Unsafe(_y.GetCoefficient(l), _X[l]->GetRotation(i), xl);
            axAdd_Unsafe(x, xl, x);
        }

        double qi[4];
        quat_Unsafe(x, qi);

        double qg[4];
        quat_Unsafe(_Xg.GetRotation(), qg);

        double q[4];
        quatMultiply_Unsafe(qg, qi, q);

        arapResiduals_ScaleV_Unsafe(_V.GetVertex(i), _V.GetVertex(j),
                                    _V1.GetVertex(i), _V1.GetVertex(j),
                                    w, q, _s.GetScale(), &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        // qi
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double w = GetEdgeWeight(k);

        // Initialise S[-1] (S.num_rows() == _n + 1)
        Matrix<double> S(_n + 1, 3, 0.);
        Matrix<double> yX(_n, 3);

        // Set remaining S
        for (int l = 0; l < _n; ++l)
        {
            axScale_Unsafe(_y.GetCoefficient(l), _X[l]->GetRotation(i), yX[l]);
            axAdd_Unsafe(S[l], yX[l], S[l+1]);
        }

        // qi
        double qi[4];
        quat_Unsafe(S[_n], qi);

        // qg
        double qg[4];
        quat_Unsafe(_Xg.GetRotation(), qg);

        // qg * qi -> q
        double q[4];
        quatMultiply_Unsafe(qg, qi, q);
        
        switch (whichParam)
        {
        case 0:
            // Xg
            {
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w, q, Jq);

                // dq/dqg
                double Dqg[16];
                quatMultiply_dp_Unsafe(qi, Dqg);

                // dqg/xg
                double Dg[12];
                quatDx_Unsafe(_Xg.GetRotation(), Dg);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqg, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Dg, J[0]);

                return;
            }
        case 1:
            // s
            {
                // arapJac_s_Unsafe(w, q, _V.GetVertex(i), _V.GetVertex(j), J[0]);
                subtractVectors_Static<double, 3>(_V1.GetVertex(i), _V1.GetVertex(j), J[0]);
                scaleVectorIP_Static<double, 3>(w, J[0]);

                return;
            }
        case 2:
            // Vi
            {
                arapJac_V_Unsafe(true, w, q, 1.0, J[0]);
                return;
            }
        case 3:
            // Vj
            {
                arapJac_V_Unsafe(false, w, q, 1.0, J[0]);
                return;
            }
        case 4:
            // V1i 
            {
                arapJac_V1_Unsafe(true, w * _s.GetScale(), J[0]);
                return;
            }
        case 5:
            // V1j
            {
                arapJac_V1_Unsafe(false, w * _s.GetScale(), J[0]);
                return;
            }
        default:
            break;
        }

        // dr/dq
        double Jq[12];
        arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w, q, Jq);

        // dq/dqi
        double Dqi[16];
        quatMultiply_dq_Unsafe(qg, Dqi);

        // dqi/xi
        double Di[12];
        quatDx_Unsafe(S[_n], Di);

        int l = whichParam - 6;
        if (l < _n)
        {
            // Xl

            // Initialise `dxdxl` to eye(3)
            double dxdxl[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};

            // Apply chain rule `l-1` times
            for (int m = _n - 1; m > l; --m)
            {
                // S[m-1], yX[m]
                double Da[9];
                axAdd_da_Unsafe(S[m], yX[m], Da);

                double A[9];
                multiply_A_B_Static<double, 3, 3, 3>(dxdxl, Da, A);
                copy(A, A+9, dxdxl);
            }

            // Apply final product
            double Db[9];
            axAdd_db_Unsafe(S[l], yX[l], Db);
            scaleVectorIP_Static<double, 9>(_y.GetCoefficient(l), Db);

            double A[12], B[9];
            multiply_A_B_Static<double, 3, 3, 3>(dxdxl, Db, A);
            copy(A, A+9, dxdxl);

            // J
            multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqi, A);
            multiply_A_B_Static<double, 3, 4, 3>(A, Di, B);
            multiply_A_B_Static<double, 3, 3, 3>(B, dxdxl, J[0]);
        }
        else
        {
            // yl
            l -= _n;

            // Initialise `dxdyl` to eye(3) (eventually only has 3 elements)
            double dxdyl[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};

            // Apply chain rule `l-1` times
            for (int m = _n - 1; m > l; --m)
            {
                // S[m-1], yX[m]
                double Da[9];
                axAdd_da_Unsafe(S[m], yX[m], Da);

                double A[9];
                multiply_A_B_Static<double, 3, 3, 3>(dxdyl, Da, A);
                copy(A, A+9, dxdyl);
            }

            // Apply final product
            double Db[9];
            axAdd_db_Unsafe(S[l], yX[l], Db);

            double A[12], B[9];
            multiply_A_B_Static<double, 3, 3, 3>(dxdyl, Db, A);
            multiply_A_v_Static<double, 3, 3>(A, _X[l]->GetRotation(i), dxdyl);

            // J
            multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqi, A);
            multiply_A_B_Static<double, 3, 4, 3>(A, Di, B);
            multiply_A_v_Static<double, 3, 3>(B, dxdyl, J[0]);
        }
    }

protected:
    const VertexNode & _V;
    const GlobalRotationNode & _Xg;
    const ScaleNode & _s;
    const vector<RotationNode *> & _X;
    const CoefficientsNode & _y;
    const VertexNode & _V1;

    const Mesh & _mesh;
    const double _w;
    bool _uniformWeights;
    int _n;
};

#endif

