#ifndef __SPILLAGE_H__
#define __SPILLAGE_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"

#include <algorithm>
using namespace std;

// SpillageEnergy
class SpillageEnergy : public Energy
{
public:
    SpillageEnergy(const VertexNode & V, const Matrix<double> & Rx, const Matrix<double> & Ry,
                   const double w)
        : _V(V), _Rx(Rx), _Ry(Ry), _w(w)
    {}

    virtual ~SpillageEnergy()
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_V.GetParamId());

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 2));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        return k + _V.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _V.GetVertices().num_rows();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        unsigned int r, c;
        VertexToRowColLookup(k, r, c);

        e[0] = _w * _Rx[r][c];
        e[1] = _w * _Ry[r][c];
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        unsigned int r, c;
        VertexToRowColLookup(k, r, c);

        J[0][0] = _w * (_Rx[r][SafeNextColumn(c)] - _Rx[r][c]);
        J[0][1] = _w * (_Rx[SafePrevRow(r)][c] - _Rx[r][c]);
        J[0][2] = 0.;
        J[1][0] = _w * (_Ry[r][SafeNextColumn(c)] - _Ry[r][c]);
        J[1][1] = _w * (_Ry[SafePrevRow(r)][c] - _Ry[r][c]);
        J[1][2] = 0.;
    }

protected:
    void VertexToRowColLookup(const int i, unsigned int & r, unsigned int & c) const
    {
        const double * Vi = _V.GetVertex(i);

        int c_ = static_cast<int>(floor(Vi[0]));
        int r_ = static_cast<int>(floor(_Rx.num_rows() - Vi[1]));

        c = static_cast<unsigned int>(max(c_, 0));
        r = static_cast<unsigned int>(max(r_, 0));

        c = min(_Rx.num_cols() - 1, c);
        r = min(_Rx.num_rows() - 1, r);
    }

    int SafePrevRow(const int r) const 
    {
        return (r - 1) >= 0 ? r - 1 : 0;
    }

    int SafeNextColumn(const int c) const 
    {
        const int N = _Rx.num_cols();
        return (c + 1) < N ? c + 1 : N - 1;
    }

    const VertexNode & _V;
    const double _w;

    const Matrix<double> & _Rx;
    const Matrix<double> & _Ry;
};

#endif
