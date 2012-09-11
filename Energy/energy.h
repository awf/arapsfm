#ifndef __ENERGY_H__
#define __ENERGY_H__

#include "Math/v3d_nonlinlsq.h"

#include <vector>
using namespace std;

// Energy
class Energy
{
public:
    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) = 0;

    virtual int GetCorrespondingParam(const int k, const int i) const = 0;
    virtual int GetNumberOfMeasurements() const = 0;

    virtual void EvaluateResidual(const int k, Vector<double> & e) const = 0;
    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const = 0;
};

// Energy_CostFunction
class Energy_CostFunction : public NLSQ_CostFunction
{
public:
    Energy_CostFunction(const Energy & parentEnergy,
                        const vector<int> * pUsedParamTypes,
                        const vector<int> * pResidualMap = nullptr)
        : _parentEnergy(parentEnergy),
          _pUsedParamTypes(pUsedParamTypes),
          _pResidualMap(pResidualMap),
          NLSQ_CostFunction(*pUsedParamTypes, 3, nullptr)
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
    }

    virtual void fillJacobian(const int whichParam, const int paramIx, const int k, Matrix<double> & Jdst, const int iteration) const
    {
        _parentEnergy.EvaluateJacobian(TranslateResidualIndex(k), whichParam, Jdst);
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
};

#endif
