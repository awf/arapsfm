#ifndef __ENERGY_H__
#define __ENERGY_H__

#include "Math/v3d_nonlinlsq.h"
#include <vector>
#include "Energy/residual.h"

using namespace std;

// Energy
class Energy
{
public:
    Energy(const double w) : _w(w) {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) = 0;

    virtual int GetCorrespondingParam(const int k, const int i) const = 0;
    virtual int GetNumberOfMeasurements() const = 0;

    virtual void EvaluateResidual(const int k, Vector<double> & e) const = 0;
    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const = 0;

    virtual bool CanBeginIteration() const { return true; }

    virtual double GetWeight() const { return _w; }

protected:
    const double _w;    
};

// Energy_CostFunction
class Energy_CostFunction : public NLSQ_CostFunction
{
public:
    Energy_CostFunction(const Energy & parentEnergy,
                        const vector<int> * pUsedParamTypes,
                        int measurementDim,
                        const vector<int> * pResidualMap = nullptr,
                        const ResidualTransform * pResidualTransform = nullptr)
        : _parentEnergy(parentEnergy),
          _pUsedParamTypes(pUsedParamTypes),
          _pResidualMap(pResidualMap),
          _pResidualTransform(pResidualTransform),
          NLSQ_CostFunction(*pUsedParamTypes, measurementDim, nullptr)
    {}
                     
    virtual ~Energy_CostFunction()
    {
        delete _pUsedParamTypes;
        if (_pResidualMap != nullptr)
            delete _pResidualMap;
    }

    virtual int correspondingParam(const int k, const int i) const
    {
        return _parentEnergy.GetCorrespondingParam(TranslateResidualIndex(k), i);
    }

    virtual int numMeasurements() const
    {
        if (_pResidualMap == nullptr)
            return _parentEnergy.GetNumberOfMeasurements();

        return _pResidualMap->size();
    }

    virtual void evalResidual(const int k, Vector<double> & e) const
    {
        _parentEnergy.EvaluateResidual(TranslateResidualIndex(k), e);

        if (_pResidualTransform != nullptr)
        {
            for (unsigned int i=0; i < e.size(); ++i)
                e[i] = _pResidualTransform->Transform(e[i]);
        }
    }

    virtual void fillJacobian(const int whichParam, const int paramIx, const int k, 
                              const Vector<double> & e, Matrix<double> & Jdst, const int iteration) const
    {
        _parentEnergy.EvaluateJacobian(TranslateResidualIndex(k), whichParam, Jdst);

        if (_pResidualTransform != nullptr)
        {
            for (unsigned int i=0; i < Jdst.num_rows(); ++i)
            {
                const double d = _pResidualTransform->Derivative(e[i]);
                for (unsigned int j=0; j < Jdst.num_cols(); ++j)
                    Jdst[i][j] *= d;
            }
        }
    }

protected:
    int TranslateResidualIndex(const int k) const
    {
        if (_pResidualMap == nullptr)
            return k;

        return (*_pResidualMap)[k];
    }

    const Energy & _parentEnergy;
    const vector<int> * _pUsedParamTypes;
    const vector<int> * _pResidualMap;
    const ResidualTransform * _pResidualTransform;
};

#endif
