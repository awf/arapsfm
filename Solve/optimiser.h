#ifndef __OPTIMISER_H__
#define __OPTIMISER_H__

#include "Solve/problem.h"

class Optimiser : NLSQ_LM_Optimizer
{
public:
    Optimiser(const NLSQ_ParamDesc & paramDesc,
              vector<NLSQ_CostFunction *> & costFunctions,
              Problem & problem)
        : NLSQ_LM_Optimizer(paramDesc, costFunctions, false),
          _problem(problem)
    {}

    virtual void updateParameters(const int paramType, const VectorArrayAdapter<double> & delta)
    {
        _problem.UpdateParameter(paramType, delta);
    }

    virtual void saveAllParameters()
    {
        _problem.Save();
    }

    virtual void restoreAllParameters()
    {
        _problem.Restore();
    }

    virtual void increaseLambda()
    {
        if (useAsymmetricLambda)
        {
            lambda *= _nu; _nu *= 2.0;
        }
        else
        {   
            lambda *= 10.0;
        }
    }

    virtual void decreaseLambda(double const rho)
    {
        if (useAsymmetricLambda)
        {
            double const r = 2*rho - 1.0;
            lambda *= std::max<double>(1.0/3.0, 1 - r*r*r);
            if (lambda < 1e-10) lambda = 1e-10;
            _nu = 2;
        }
        else
        {
            lambda = std::max(1e-10, lambda * 0.1); 
        }
    }

    bool useAsymmetricLambda;


protected:
    Problem & _problem;
};

#endif
