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
        : _V(V), _X(X), _V1(V1), _mesh(mesh)
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
    
};

// RigidTransformARAPEnergy
class RigidTransformARAPEnergy: public Energy
{
public:
    RigidTransformARAPEnergy(const VertexNode & V,  const RotationNode & X, 
                             const RotationNode & Xg, const ScaleNode & s,
                             const VertexNode & V1, const Mesh & mesh, const double w,
                             bool uniformWeights)
        : Energy(w), _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), _uniformWeights(uniformWeights)
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
    bool _uniformWeights;
};

// RigidTransformARAPEnergy2 (uses `arapResiduals_ScaleV_Unsafe` and `GlobalRotationNode`)
class RigidTransformARAPEnergy2: public Energy
{
public:
    RigidTransformARAPEnergy2(const VertexNode & V, const GlobalRotationNode & Xg, const ScaleNode & s,
                              const RotationNode & X, const VertexNode & V1, const Mesh & mesh, const double w,
                              bool uniformWeights, bool fixedScale)
        : Energy(w), _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), 
          _uniformWeights(uniformWeights), _fixedScale(fixedScale)
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
            // V1i
            {
                
                arapJac_V1_Unsafe(true, w * _s.GetScale(), J[0]);
                return;
            }
        case 3:
            // V1j
            {
                arapJac_V1_Unsafe(false, w * _s.GetScale(), J[0]);
                return;
            }

        case 4:
            // s
            {
                subtractVectors_Static<double, 3>(_V1.GetVertex(i), _V1.GetVertex(j), J[0]);
                scaleVectorIP_Static<double, 3>(w, J[0]);
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
    bool _uniformWeights;
    bool _fixedScale;
};

// RigidTransformARAPEnergy2B (uses `arapResiduals_ScaleV_Unsafe` and
// `GlobalRotationNode` but `V` is free instead of `V1`)
class RigidTransformARAPEnergy2B: public Energy
{
public:
    RigidTransformARAPEnergy2B(const VertexNode & V, const GlobalRotationNode & Xg, const ScaleNode & s,
                               const RotationNode & X, const VertexNode & V1, const Mesh & mesh, const double w,
                               bool uniformWeights, bool fixedScale)
        : Energy(w), _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), 
          _uniformWeights(uniformWeights), _fixedScale(fixedScale)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_Xg.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
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
            return _mesh.GetHalfEdge(k, 0) + _V.GetOffset();
        case 3:
            return _mesh.GetHalfEdge(k, 1) + _V.GetOffset();
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
            // s
            {
                subtractVectors_Static<double, 3>(_V1.GetVertex(i), _V1.GetVertex(j), J[0]);
                scaleVectorIP_Static<double, 3>(w, J[0]);
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
    bool _uniformWeights;
    bool _fixedScale;
};

// RigidTransformARAPEnergy3 (uses `GlobalRotationNode`)
class RigidTransformARAPEnergy3: public Energy
{
public:
    RigidTransformARAPEnergy3(const VertexNode & V, const GlobalRotationNode& Xg, const ScaleNode & s,
                              const RotationNode & X, const VertexNode & V1, const Mesh & mesh, const double w,
                              bool uniformWeights, bool fixedScale)
        : Energy(w), _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), 
          _uniformWeights(uniformWeights), _fixedScale(fixedScale)
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
        quat_Unsafe(_Xg.GetRotation(), qg);

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
                quatDx_Unsafe(_Xg.GetRotation(), Dg);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqg, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Dg, J[0]);

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
    const RotationNode & _X;
    const GlobalRotationNode & _Xg;
    const ScaleNode & _s;
    const VertexNode & _V1;

    const Mesh & _mesh;
    bool _uniformWeights;
    bool _fixedScale;
};

// RigidTransformARAPEnergy3B (uses `GlobalRotationNode` and `V` is free
// instead of `V1`)
class RigidTransformARAPEnergy3B: public Energy
{
public:
    RigidTransformARAPEnergy3B(const VertexNode & V, const GlobalRotationNode& Xg, const ScaleNode & s,
                               const RotationNode & X, const VertexNode & V1, const Mesh & mesh, const double w,
                               bool uniformWeights, bool fixedScale)
        : Energy(w), _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), 
          _uniformWeights(uniformWeights), _fixedScale(fixedScale)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_X.GetParamId());
        pUsedParamTypes->push_back(_Xg.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
        pUsedParamTypes->push_back(_V.GetParamId());
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
            return _mesh.GetHalfEdge(k, 0) + _V.GetOffset();
        case 3:
            return _mesh.GetHalfEdge(k, 1) + _V.GetOffset();
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
        quat_Unsafe(_Xg.GetRotation(), qg);

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
                quatDx_Unsafe(_Xg.GetRotation(), Dg);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqg, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Dg, J[0]);

                return;
            }

        case 2:
            // Vi
            {
                arapJac_V_Unsafe(true, w * _s.GetScale(), q, 1.0, J[0]);
                return;
            }
        case 3:
            // Vj
            {
                arapJac_V_Unsafe(false, w * _s.GetScale(), q, 1.0, J[0]);
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
    const RotationNode & _X;
    const GlobalRotationNode & _Xg;
    const ScaleNode & _s;
    const VertexNode & _V1;

    const Mesh & _mesh;
    bool _uniformWeights;
    bool _fixedScale;
};

// TODO DualArapEnergy::EvaluateJacobian DOES NOT take into account the cotangent weight
// dependence

// DualARAPEnergy
class DualARAPEnergy : public Energy
{
public:
    DualARAPEnergy(const VertexNode & V,  const RotationNode & X, const VertexNode & V1,
               const Mesh & mesh, const double w, bool uniformWeights = true)
        : Energy(w), _V(V), _X(X), _V1(V1), _mesh(mesh), _uniformWeights(uniformWeights) 
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
    

    bool _uniformWeights;
};

// DualScaledARAPEnergy
class DualScaledARAPEnergy : public Energy
{
public:
    DualScaledARAPEnergy(const VertexNode & V,  const RotationNode & X, const ScaleNode & s,
                         const VertexNode & V1, const Mesh & mesh, const double w, 
                         bool uniformWeights = true)
        : Energy(w), _V(V), _X(X), _s(s), _V1(V1), _mesh(mesh), _uniformWeights(uniformWeights) 
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
        : Energy(w), _V(V), _X(X), _Xg(Xg), _s(s), _V1(V1), _mesh(mesh), _uniformWeights(uniformWeights)
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
        : Energy(w), _V(V), _Xg(Xg), _s(s), _X(X), _y(y), _V1(V1), _mesh(mesh), _uniformWeights(uniformWeights), _n(X.size())
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
    bool _uniformWeights;
    int _n;
};

// SectionedBasisArapEnergy
class SectionedBasisArapEnergy : public Energy
{
public:
    SectionedBasisArapEnergy(const VertexNode & V, const GlobalRotationNode & Xg, const ScaleNode & s,
                             const RotationNode & Xb, const CoefficientsNode & y, 
                             const RotationNode & X, const VertexNode & V1, 
                             const Matrix<int> & K, const Mesh & mesh, const double w,
                             bool uniformWeights, 
                             bool fixedXb, bool fixedV, bool fixedV1, bool fixedScale)
        : Energy(w), _V(V), _Xg(Xg), _s(s), _Xb(Xb), _y(y), _X(X), _V1(V1), _K(K), _mesh(mesh), 
          _uniformWeights(uniformWeights), 
          _fixedXb(fixedXb), _fixedV(fixedV), _fixedV1(fixedV1), _fixedScale(fixedScale)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * fixedRotResidualMap = new vector<int>;
        vector<int> * basisRotResidualMap = new vector<int>;
        vector<int> * freeRotResidualMap = new vector<int>;

        for (int k=0; k < _mesh.GetNumberOfHalfEdges(); k++)
        {
            int i = _mesh.GetHalfEdge(k, 0);

            if (_K[i][0] < 0)
            {
                // free (independent) rotation
                freeRotResidualMap->push_back(k);
            }
            else if (_K[i][0] == 0)
            {
                // fixed rotation
                fixedRotResidualMap->push_back(k);
            }
            else
            {
                // single-axis (basis) rotation
                basisRotResidualMap->push_back(k);
            }
        }

        // construct cost functions
        if (freeRotResidualMap->size() > 0)
        {
            // free (independent) rotation
            vector<int> * pUsedParamTypes = new vector<int>;
            pUsedParamTypes->push_back(_Xg.GetParamId());
            if (!_fixedV1)
            {
                pUsedParamTypes->push_back(_V1.GetParamId());
                pUsedParamTypes->push_back(_V1.GetParamId());
            }
            pUsedParamTypes->push_back(_X.GetParamId());
            if (!_fixedV)
            {
                pUsedParamTypes->push_back(_V.GetParamId());
                pUsedParamTypes->push_back(_V.GetParamId());
            }
            if (!_fixedScale)
                pUsedParamTypes->push_back(_s.GetParamId());

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, freeRotResidualMap));
        }

        if (fixedRotResidualMap->size() > 0)
        {
            // fixed rotation
            vector<int> * pUsedParamTypes = new vector<int>;
            pUsedParamTypes->push_back(_Xg.GetParamId());
            if (!_fixedV1)
            {
                pUsedParamTypes->push_back(_V1.GetParamId());
                pUsedParamTypes->push_back(_V1.GetParamId());
            }
            if (!_fixedV)
            {
                pUsedParamTypes->push_back(_V.GetParamId());
                pUsedParamTypes->push_back(_V.GetParamId());
            }
            if (!_fixedScale)
                pUsedParamTypes->push_back(_s.GetParamId());

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, fixedRotResidualMap));
        }

        if (basisRotResidualMap->size() > 0)
        {
            // single-axis (basis) rotation
            vector<int> * pUsedParamTypes = new vector<int>;
            pUsedParamTypes->push_back(_Xg.GetParamId());
            if (!_fixedV1)
            {
                pUsedParamTypes->push_back(_V1.GetParamId());
                pUsedParamTypes->push_back(_V1.GetParamId());
            }
            if (!_fixedXb)
                pUsedParamTypes->push_back(_Xb.GetParamId());
            pUsedParamTypes->push_back(_y.GetParamId());
            if (!_fixedV)
            {
                pUsedParamTypes->push_back(_V.GetParamId());
                pUsedParamTypes->push_back(_V.GetParamId());
            }
            if (!_fixedScale)
                pUsedParamTypes->push_back(_s.GetParamId());

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, basisRotResidualMap));
        }
    }

    virtual int GetCorrespondingParam(const int k, const int l) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);

        // parameters common to all three types of cost functions
        int m = 0;

        if (l == m++) 
            return _Xg.GetOffset();
        else if (!_fixedV1 && (l == m++)) 
            return i + _V1.GetOffset();
        else if (!_fixedV1 && (l == m++)) 
            return j + _V1.GetOffset();

        if (_K[i][0] < 0)
        {
            // free (independent) rotation
            if (l == m++) 
                return _K[i][1] + _X.GetOffset();
        }
        else if (_K[i][0] == 0)
        {
            // fixed rotation
        }
        else
        {
            // single-axis (basis) rotation
            if (!_fixedXb && (l == m++)) 
                return _K[i][1] + _Xb.GetOffset();
            else if (l == m++) 
                return _K[i][0] - 1 + _y.GetOffset();
        }

        if (!_fixedV && (l == m++)) 
            return i + _V.GetOffset();
        else if (!_fixedV && (l == m++)) 
            return j + _V.GetOffset();
        else if (!_fixedScale && (l == m++)) 
            return _s.GetOffset();

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

        double qi[4] = {0., 0., 0., 1.};

        if (_K[i][0] < 0)
        {
            quat_Unsafe(_X.GetRotation(_K[i][1]), qi);
        }
        else if (_K[i][0] > 0)
        {
            double x[3];
            axScale_Unsafe(_y.GetCoefficient(_K[i][0] - 1), _Xb.GetRotation(_K[i][1]), x);
            quat_Unsafe(x, qi);
        }

        double qg[4];
        quat_Unsafe(_Xg.GetRotation(), qg);

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

        double qi[4] = {0., 0., 0., 1.};

        if (_K[i][0] < 0)
        {
            quat_Unsafe(_X.GetRotation(_K[i][1]), qi);
        }
        else if (_K[i][0] > 0)
        {
            double x[3];
            axScale_Unsafe(_y.GetCoefficient(_K[i][0] - 1), _Xb.GetRotation(_K[i][1]), x);
            quat_Unsafe(x, qi);
        }

        double qg[4];
        quat_Unsafe(_Xg.GetRotation(), qg);

        double q[4];
        quatMultiply_Unsafe(qg, qi, q);

        int m = 0;
        
        if (whichParam == m++)
        {
            // Xg
            // dr/dq
            double Jq[12];
            arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

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
        else if (!_fixedV1 && (whichParam == m++))
        {
            // V1i
            arapJac_V1_Unsafe(true, w, J[0]);
            return;
        }
        else if (!_fixedV1 && (whichParam == m++))
        {
            // V1j
            arapJac_V1_Unsafe(false, w, J[0]);
            return;
        }

        if (_K[i][0] < 0)
        {
            // free (independent) rotation
            if (whichParam == m++)
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
                quatDx_Unsafe(_X.GetRotation(_K[i][1]), Di);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqi, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Di, J[0]);

                return;
            }
        }
        else if (_K[i][0] == 0)
        {
            // fixed rotation
        }
        else
        {
            // single-axis (basis) rotation
            if (!_fixedXb && (whichParam == m++))
            {
                // Xb
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                // dq/dqi
                double Dqi[16];
                quatMultiply_dq_Unsafe(qg, Dqi);

                // dqi/xi
                double x[3];
                axScale_Unsafe(_y.GetCoefficient(_K[i][0] - 1), _Xb.GetRotation(_K[i][1]), x);

                double Di[12];
                quatDx_Unsafe(x, Di);

                double A[12];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqi, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Di, J[0]);

                // dxi/dxb
                scaleVectorIP_Static<double, 9>(_y.GetCoefficient(_K[i][0] - 1), J[0]);
                return;
            }
            else if (whichParam == m++)
            {
                // y
                // dr/dq
                double Jq[12];
                arapJac_Q_Unsafe(_V.GetVertex(i), _V.GetVertex(j), w * _s.GetScale(), q, Jq);

                // dq/dqi
                double Dqi[16];
                quatMultiply_dq_Unsafe(qg, Dqi);

                // dqi/xi
                double x[3];
                axScale_Unsafe(_y.GetCoefficient(_K[i][0] - 1), _Xb.GetRotation(_K[i][1]), x);

                double Di[12];
                quatDx_Unsafe(x, Di);

                double A[12], B[9];
                multiply_A_B_Static<double, 3, 4, 4>(Jq, Dqi, A);
                multiply_A_B_Static<double, 3, 4, 3>(A, Di, B);

                // dxi/y
                multiply_A_v_Static<double, 3, 3>(B, _Xb.GetRotation(_K[i][1]), J[0]);
                return;
            }
        }

        if (!_fixedV && (whichParam == m++)) 
        {
            // Vi
            arapJac_V_Unsafe(true, w * _s.GetScale(), q, 1.0, J[0]);
            return;
        }
        else if (!_fixedV && (whichParam == m++)) 
        {
            // Vj
            arapJac_V_Unsafe(false, w * _s.GetScale(), q, 1.0, J[0]);
            return;
        }
        else if (!_fixedScale && (whichParam == m++)) 
        {
            // s
            arapJac_s_Unsafe(w, q, _V.GetVertex(i), _V.GetVertex(j), J[0]);
            return;
        }

        assert(false);
    }

protected:
    const VertexNode & _V;
    const GlobalRotationNode & _Xg;
    const ScaleNode & _s;
    const RotationNode & _Xb;
    const CoefficientsNode & _y;
    const RotationNode & _X;
    const VertexNode & _V1;

    const Matrix<int> & _K;

    const Mesh & _mesh;
    bool _uniformWeights;
    bool _fixedXb;
    bool _fixedV;
    bool _fixedV1;
    bool _fixedScale;
};

// RotationRegulariseEnergy
class RotationRegulariseEnergy : public Energy
{
public:
    RotationRegulariseEnergy(const RotationNode & X, const double w)
        : Energy(w), _X(X)
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
    
};

// GlobalRotationRegulariseEnergy
class GlobalRotationRegulariseEnergy : public Energy
{
public:
    GlobalRotationRegulariseEnergy(const GlobalRotationNode & X, const double w)
        : Energy(w), _X(X)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>(1, _X.GetParamId());
        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        return _X.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        return 1;
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        const double * X = _X.GetRotation();

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
    const GlobalRotationNode & _X;
    
};

// SectionedRotationsVelocityEnergy
class SectionedRotationsVelocityEnergy : public Energy
{
public:
    SectionedRotationsVelocityEnergy(const GlobalRotationNode & Xg0, const CoefficientsNode & y0, const RotationNode & X0,
                                     const GlobalRotationNode & Xg, const CoefficientsNode & y, const RotationNode & X,
                                     const RotationNode & Xb, const Matrix<int> & K, const double w,
                                     bool fixed0, bool fixedXb)
        : _Energy(w), Xg0(Xg0), _y0(y0), _X0(X0), _Xg(Xg), _y(y), _X(X), _Xb(Xb), _K(K), _fixed0(fixed0), _fixedXb(fixedXb),
          _numMeasurements(0)
    {
        for (int k=0; k < _K.num_rows(); ++k)
        {
            if (_K[k][0] != 0)
                _numMeasurements++;
        }
    }

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * basisRotResidualMap = new vector<int>;
        vector<int> * freeRotResidualMap = new vector<int>;

        for (int k=0; k < _K.num_rows(); ++k)
        {
            if (_K[k][0] < 0)
            {
                // free (independent) rotation
                freeRotResidualMap->push_back(k);
            }
            else if (_K[k] > 0)
            {
                // single-axis (basis) rotation
                basisRotResidualMap->push_back(k);
            }
        }

        // construct cost functions
        if (freeRotResidualMap->size() > 0)
        {
            // free (independent) rotation
            vector<int> * pUsedParamTypes = new vector<int>;
            pUsedParamTypes->push_back(_Xg.GetParamId());
            pUsedParamTypes->push_back(_X.GetParamId());
            if (!_fixed0)
            {
                pUsedParamTypes->push_back(_Xg0.GetParamId());
                pUsedParamTypes->push_back(_X0.GetParamId());
            }

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, freeRotResidualMap));
        }

        if (basisRotResidualMap->size() > 0)
        {
            // single-axis (basis) rotation
            vector<int> * pUsedParamTypes = new vector<int>;
            pUsedParamTypes->push_back(_Xg.GetParamId());
            if (!_fixedXb)
                pUsedParamTypes->push_back(_Xb.GetParamId());

            pUsedParamTypes->push_back(_y.GetParamId());

            if (!_fixed0)
            {
                pUsedParamTypes->push_back(_Xg0.GetParamId());
                pUsedParamTypes->push_back(_y0.GetParamId());
            }

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, basisRotResidualMap));
        }
    }
                                     
    virtual int GetCorrespondingParam(const int k, const int l) const
    {
        int m = 0;
        if (l == m++)
            return _Xg.GetOffset();

        if (_K[k][0] < 0)
        {
            // free (independent) rotation
            if (l == m++)
                return _K[k][1] + _X.GetOffset();
        }
        else
        {
            // single-axis (basis) rotation
            if (!_fixedXb && (l == m++))
                return _K[k][1] + _Xb.GetOffset();
            else if (l == m++)
                return _K[k][0] - 1 + _y.GetOffset();
        }

        if (!_fixed0)
        {
            if (l == m++)
                return _Xg0.GetOffset();

            if (_K[k][0] < 0)
            {
                // free (independent) rotation
                if (l == m++)
                    return _K[k][1] + _X0.GetOffset();
            }
            else
            {
                // single-axis (basis) rotation
                if (l == m++)
                    return _K[k][0] - 1 + _y0.GetOffset();
            }
        }

        assert(false);

        return -1;
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _numMeasurements;
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        double x0i[3], xi[3];

        if (_K[k][0] < 0)
        {
            const double * x0i_ = _X0.GetRotation(_K[k][1]);
            copy(x0i_, x0i_ + 3, x0i);

            const double * xi_ = _X.GetRotation(_K[k][1]);
            copy(xi_, xi_ + 3, xi);
        }
        else if (_K[k][0] == 0)
        {
            assert(false);
        }
        else
        {
            // _K[k][0] > 0
            axScale_Unsafe(_y0.GetCoefficient(_K[k][0] - 1), _Xb.GetRotation(_K[k][1]), x0i);
            axScale_Unsafe(_y.GetCoefficient(_K[k][0] - 1), _Xb.GetRotation(_K[k][1]), xi);
        }

        double x0[3];
        axAdd_Unsafe(_Xg0.GetRotation(), x0i, x0);

        double x[3];
        axAdd_Unsafe(_Xg.GetRotation(), xi, x);

        double inv_x0[3];
        axScale_Unsafe(-1.0, x0, inv_x0);

        axAdd_Unsafe(inv_x0, x, e.begin());
        scaleVectorIP_Static<double, 3>(_w, e.begin());
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        double x0i[3], xi[3];

        if (_K[k][0] < 0)
        {
            const double * x0i_ = _X0.GetRotation(_K[k][1]);
            copy(x0i_, x0i_ + 3, x0i);

            const double * xi_ = _X.GetRotation(_K[k][1]);
            copy(xi_, xi_ + 3, xi);
        }
        else if (_K[k][0] == 0)
        {
            assert(false);
        }
        else
        {
            // _K[k][0] > 0
            axScale_Unsafe(_y0.GetCoefficient(_K[k][0] - 1), _Xb.GetRotation(_K[k][1]), x0i);
            axScale_Unsafe(_y.GetCoefficient(_K[k][0] - 1), _Xb.GetRotation(_K[k][1]), xi);
        }

        double x0[3];
        axAdd_Unsafe(_Xg0.GetRotation(), x0i, x0);

        double x[3];
        axAdd_Unsafe(_Xg.GetRotation(), xi, x);

        double inv_x0[3];
        axScale_Unsafe(-1.0, x0, inv_x0);

        // Jx
        double Jx[9];
        axAdd_db_Unsafe(inv_x0, x, Jx);
        scaleVectorIP_Static<double, 9>(_w, Jx);

        // Jninv
        double Jninv[9];
        axAdd_da_Unsafe(inv_x0, x, Jninv);
        scaleVectorIP_Static<double, 9>(-_w, Jninv);

        int m = 0;
        if (whichParam == m++)
        {
            // Xg
            // Jxg
            double Jxg[9];
            axAdd_da_Unsafe(_Xg.GetRotation(), xi, Jxg);

            multiply_A_B_Static<double, 3, 3, 3>(Jx, Jxg, J[0]);
            return;
        }

        // Xi
        double Jxi[9];
        axAdd_db_Unsafe(_Xg.GetRotation(), xi, Jxi);

        // X0i
        double Jx0i[9];
        axAdd_db_Unsafe(_Xg0.GetRotation(), x0i, Jx0i);

        if (_K[k][0] < 0)
        {
            // free (independent) rotation
            if (whichParam == m++)
            {
                // X
                multiply_A_B_Static<double, 3, 3, 3>(Jx, Jxi, J[0]);
                return;
            }
        }
        else
        {
            // single-axis (basis) rotation
            if (!_fixedXb && (whichParam == m++))
            {
                // Xb
                double Jl[9], Jr[9];

                scaleVectorIP_Static<double, 9>(_y0.GetCoefficient(_K[k][0] - 1), Jx0i);
                multiply_A_B_Static<double, 3, 3, 3>(Jninv, Jx0i, Jl);

                scaleVectorIP_Static<double, 9>(_y.GetCoefficient(_K[k][0] - 1), Jxi);
                multiply_A_B_Static<double, 3, 3, 3>(Jx, Jxi, Jr);

                fillVector_Static<double, 9>(0., J[0]);
                addVectors_Static<double, 9>(Jl, Jr, J[0]);
                return;
            }
            else if (whichParam == m++)
            {
                // y
                double JXby[9];
                multiply_A_B_Static<double, 3, 3, 3>(Jx, Jxi, JXby);
                multiply_A_v_Static<double, 3, 3>(JXby, _Xb.GetRotation(_K[k][1]), J[0]);
                return;
            }
        }

        if (!_fixed0)
        {
            if (whichParam == m++)
            {
                // Xg0
                double Jxg0[9];
                axAdd_da_Unsafe(_Xg0.GetRotation(), x0i, Jxg0);

                multiply_A_B_Static<double, 3, 3, 3>(Jninv, Jxg0, J[0]);
                return;
            }

            if (_K[k][0] < 0)
            {
                // free (independent) rotation
                if (whichParam == m++)
                {
                    // X0
                    multiply_A_B_Static<double, 3, 3, 3>(Jninv, Jx0i, J[0]);
                    return;
                }
            }
            else
            {
                if (whichParam == m++)
                {
                    // y0
                    double JXby0[9];
                    multiply_A_B_Static<double, 3, 3, 3>(Jninv, Jx0i, JXby0);
                    multiply_A_v_Static<double, 3, 3>(JXby0, _Xb.GetRotation(_K[k][1]), J[0]);
                    return;
                }
            }
        }

        assert(false);
        return;
    }

protected:
    const GlobalRotationNode & _Xg0;
    const CoefficientsNode & _y0;
    const RotationNode & _X0;
    const GlobalRotationNode & _Xg;
    const CoefficientsNode & _y;
    const RotationNode & _X;
    const RotationNode & _Xb;
    const Matrix<int> & _K;
    bool _fixed0;
    bool _fixedXb;

    int _numMeasurements;
};

// GlobalRotationsDifferenceEnergy
class GlobalRotationsDifferenceEnergy : public Energy
{
public:
    GlobalRotationsDifferenceEnergy(const GlobalRotationNode & Xg, const GlobalRotationNode & Xg0, 
                                    const double w, bool fixed0)
        : Energy(w), _Xg(Xg), _Xg0(Xg0), _fixed0(fixed0)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        auto pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_Xg.GetParamId());

        if (!_fixed0)
            pUsedParamTypes->push_back(_Xg.GetParamId());
        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int l) const
    {
        switch (l)
        {
        case 0:
            return _Xg.GetOffset();
        case 1:
            return _Xg0.GetOffset();
        }

        assert(false);

        return -1;
    }

    virtual int GetNumberOfMeasurements() const 
    {
        return 1;
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        double inv_x0[3];
        axScale_Unsafe(-1.0, _Xg0.GetRotation(), inv_x0);

        axAdd_Unsafe(inv_x0, _Xg.GetRotation(), e.begin());
        scaleVectorIP_Static<double, 3>(_w, e.begin());
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        double inv_x0[3];
        axScale_Unsafe(-1.0, _Xg0.GetRotation(), inv_x0);

        if (whichParam == 0)
        {
            axAdd_db_Unsafe(inv_x0, _Xg.GetRotation(), J[0]);
            scaleVectorIP_Static<double, 9>(_w, J[0]);
            return;
        }
        else
        {
            axAdd_da_Unsafe(inv_x0, _Xg.GetRotation(), J[0]);
            scaleVectorIP_Static<double, 9>(-_w, J[0]);
            return;
        }
    }

protected:
    const GlobalRotationNode & _Xg;
    const GlobalRotationNode & _Xg0;
    bool _fixed0;
};

// GlobalScaleRegulariseEnergy
class GlobalScaleRegulariseEnergy : public Energy
{
public:
    GlobalScaleRegulariseEnergy(const ScaleNode & s, const double w)
        : _Energy(w), s(s)
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
    
};

#endif

