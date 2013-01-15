#ifndef __LINEAR_BASIS_SHAPE_PROJECTION__
#define __LINEAR_BASIS_SHAPE_PROJECTION__

#include "linear_basis_shape.h"
#include "Energy/projection.h"

// LinearBasisShapeProjectionEnergy
class LinearBasisShapeProjectionEnergy : public ProjectionEnergy
{
public:
    LinearBasisShapeProjectionEnergy(LinearBasisShapeNode & V, 
          const Vector<int> & C, const Matrix<double> & P, const double w)
        : ProjectionEnergy(V, C, P, w), __V(V)
    {} 

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        auto pUsedParamTypes = new vector<int>;
        __V.AddVertexUsedParamTypes(pUsedParamTypes);
        __V.AddGlobalUsedParamTypes(pUsedParamTypes);

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 2));
    }

    virtual int GetCorrespondingParam(const int k, const int l) const
    {
        // GetGlobalParam and adjust `whichParam` to local
        int whichParam = l;

        int p = __V.GetVertexParam(whichParam, _C[k]);
        if (p >= 0) return p;

        p = __V.GetScaleParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetGlobalRotationParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetDisplacmentParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetCoefficientParam(whichParam);
        if (p >= 0) return p;

        assert(false);
    }

    virtual void EvaluateJacobian(const int k, const int l, Matrix<double> & J) const
    {
        int whichParam = l;

        Matrix<double> Jr(2, 3);
        ProjectionEnergy::EvaluateJacobian(k, 0, Jr);

        if (__V.GetVertexParam(whichParam, _C[k]) >= 0)
        {
            Matrix<double> JV(3,3);
            __V.VertexJacobian(whichParam, _C[k], JV);
            multiply_A_B(Jr, JV, J);
            return;
        }
        
        if (__V.GetScaleParam(whichParam) >= 0)
        {
            Matrix<double> Js(3,1);
            __V.ScaleJacobian(_C[k], Js);
            multiply_A_B(Jr, Js, J);
            return;
        }
        
        if (__V.GetGlobalRotationParam(whichParam) >= 0)
        {
            Matrix<double> JXg(3,3);
            __V.GlobalRotationJacobian(_C[k], JXg);
            multiply_A_B(Jr, JXg, J);
            return;
        }
        
        if (__V.GetDisplacmentParam(whichParam) >= 0)
        {
            Matrix<double> JVd(3,3);
            __V.DisplacementJacobian(JVd);
            multiply_A_B(Jr, JVd, J);
            return;
        }
        
        if (__V.GetCoefficientParam(whichParam) >= 0)
        {
            Matrix<double> Jy(3,1);
            __V.CoefficientJacobian(whichParam, _C[k], Jy);
            multiply_A_B(Jr, Jy, J);
            return;
        }

        assert(false);
    }

protected:
    LinearBasisShapeNode & __V;
};

#endif
