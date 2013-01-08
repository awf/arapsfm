#ifndef __PROJECTION_H__
#define __PROJECTION_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"

// projResiduals_Unsafe
inline void projResiduals_Unsafe(const double * V1i, const double * Pi, const double & w, double * e)
{
    e[0] = w*(Pi[0] - V1i[0]);
    e[1] = w*(Pi[1] - V1i[1]);
}

// projJac_V_Unsafe
inline void projJac_V_Unsafe(const double & w, double * J)
{
    J[0] = -w; 
    J[1] = 0; 
    J[2] = 0; 
    J[3] = 0; 
    J[4] = -w; 
    J[5] = 0; 
}

// ProjectionEnergy
class ProjectionEnergy : public Energy
{
public:
    ProjectionEnergy(const VertexNode & V, const Vector<int> & C, const Matrix<double> & P,
                     const double w)
        : Energy(w), _V(V), _C(C), _P(P) 
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>(1, _V.GetParamId());
        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 2));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        return _C[k] + _V.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _C.size();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        int i = _C[k];

        projResiduals_Unsafe(_V.GetVertex(i), _P[k], _w, e.begin());
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        projJac_V_Unsafe(_w, J[0]);
    }

protected:
    const VertexNode & _V;

    const Vector<int> & _C;
    const Matrix<double> & _P;

    
};

// absPosResiduals_Unsafe
inline void absPosResiduals_Unsafe(const double * V1i, const double * Pi, const double & w, double * e)
{
    e[0] = w*(Pi[0] - V1i[0]);
    e[1] = w*(Pi[1] - V1i[1]);
    e[2] = w*(Pi[2] - V1i[2]);
}

// absPosJac_V_Unsafe
inline void absPosJac_V_Unsafe(const double & w, double * J)
{
    J[0] = -w; 
    J[1] = 0; 
    J[2] = 0; 
    J[3] = 0; 
    J[4] = -w; 
    J[5] = 0; 
    J[6] = 0; 
    J[7] = 0; 
    J[8] = -w; 
}

// AbsolutePositionEnergy
class AbsolutePositionEnergy : public Energy
{
public:
    AbsolutePositionEnergy(const VertexNode & V, const Vector<int> & C, const Matrix<double> & P,
                           const double w)
        : Energy(w), _V(V), _C(C), _P(P)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>(1, _V.GetParamId());
        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        return _C[k] + _V.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _C.size();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        int i = _C[k];

        absPosResiduals_Unsafe(_V.GetVertex(i), _P[k], _w, &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        absPosJac_V_Unsafe(_w, J[0]);
    }

protected:
    const VertexNode & _V;

    const Vector<int> & _C;
    const Matrix<double> & _P;

    
};


#endif
