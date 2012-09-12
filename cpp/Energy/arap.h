#ifndef __ARAP_H__
#define __ARAP_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/mesh.h"
#include "Geometry/quaternion.h"

#include <cmath>

// arapResiduals_Unsafe
inline void arapResiduals_Unsafe(const double * Vi, const double * Vj,
           const double * V1i, const double * V1j,
           const double & w, const double * q, 
           double * e)
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

    e[0] = w*(V1i[0] - V1j[0] - x0*(2*x6 + 2*x7) - x1*(-2*x4 + 2*x5) - x2*(-2*x10 - 2*x3 + 1));
    e[1] = w*(V1i[1] - V1j[1] - x0*(2*x8 - 2*x9) - x1*(-2*x11 - 2*x3 + 1) - x2*(2*x4 + 2*x5));
    e[2] = w*(V1i[2] - V1j[2] - x0*(-2*x10 - 2*x11 + 1) - x1*(2*x8 + 2*x9) - x2*(2*x6 - 2*x7));
    // END SymPy
}

// arapJac_Q_Unsafe
inline void arapJac_Q_Unsafe(const double * Vi, const double * Vj,
                 const double & w, const double * q, 
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
                 const double & w, const double * q, const double * D, 
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
inline void arapJac_V1_Unsafe(bool isV1i, const double & w, double * J)
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
inline void arapJac_V_Unsafe(bool isVi, const double & w, const double * q, double * J)
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
    scaleVectorIP_Static<double, 9>(w_, J);
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
                             w, q, &e[0]);
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

#endif

