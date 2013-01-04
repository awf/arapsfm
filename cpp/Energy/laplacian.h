#ifndef __LAPLACIAN_H__
#define __LAPLACIAN_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/mesh.h"

#include <algorithm>
using namespace std;

// LaplacianEnergy
class LaplacianEnergy : public Energy
{
public:
    LaplacianEnergy(const VertexNode & V, const vector<const ScaleNode *> && s, const Mesh & mesh, const double w)
        : _V(V), _s(s), _mesh(mesh), _w(w)
    {
        assert(_s.size() > 0);
    }

    virtual ~LaplacianEnergy()
    {
        for (int i = 0; i < _allOneRings.size(); i++)
            if (_allOneRings[i] != nullptr) delete _allOneRings[i];
    }
    
    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) 
    {
        // map size of "inclusive one ring" to vertex id
        map<int, vector<int> *> mapIncOneRingToVertex;

        for (int i=0; i < _mesh.GetNumberOfVertices(); i++)
        {
            vector<int> * oneRing = new vector<int>(_mesh.GetNRing(i, 1, true));
            _allOneRings.push_back(oneRing);
            const int n = oneRing->size();

            auto j = mapIncOneRingToVertex.find(n);
            if (j == mapIncOneRingToVertex.end())
            {
                vector<int> * v = new vector<int>(1, i);
                mapIncOneRingToVertex.insert(pair<int, vector<int> * >(n, v));
            }
            else
            {
                j->second->push_back(i);
            }
        }

        // for each size of one ring create a cost function
        for (auto i = mapIncOneRingToVertex.begin(); i != mapIncOneRingToVertex.end(); i++)
        {
            auto pUsedParamTypes = new vector<int>(i->first + _s.size());

            fill_n(pUsedParamTypes->begin(), i->first, _V.GetParamId());
            for (int j = 0; j < _s.size(); j++)
                (*pUsedParamTypes)[j + i->first] = _s[j]->GetParamId();

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, i->second));
        }
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        const vector<int> & oneRing = *_allOneRings[k];
        if (i < oneRing.size())
            return oneRing[i] + _V.GetOffset();

        int scaleIndex = i - oneRing.size();
        return _s[scaleIndex]->GetOffset();
    }

    virtual int GetNumberOfMeasurements() const 
    {
        return _mesh.GetNumberOfVertices();
    }

    virtual double GetMeanScale() const
    {
        // get the mean scale
        double sum_s = 0.;
        for (int i=0; i < _s.size(); i++)
            sum_s += _s[i]->GetScale();

        assert(sum_s > 0.);

        return sum_s / _s.size();
    }

    virtual void GetCentroid(int k, double * centroid) const
    {
        const vector<int> & oneRing = *_allOneRings[k];

        fillVector_Static<double, 3>(0., centroid);

        for (int i=1; i < oneRing.size(); i++)
        {
            makeInterpolatedVector_Static<double, 3>(1.0, centroid, 
                                                     1.0, _V.GetVertex(oneRing[i]), 
                                                     centroid);
        }

        scaleVectorIP_Static<double, 3>(1.0 / (oneRing.size() - 1), centroid);
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        // calculate the centroid
        double centroid[3];
        GetCentroid(k, centroid);

        // calulate the weighted error
        const double w = _w * GetMeanScale();

        makeInterpolatedVector_Static<double, 3>(w, _V.GetVertex(k), -w, centroid, &e[0]);
    }
    
    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        // get the indices for the (directional) incident edges
        if (whichParam < _allOneRings[k]->size())
        {
            double w = _w * GetMeanScale();

            if (whichParam > 0)
                w *= -1.0 / (_allOneRings[k]->size() - 1);

            fillVector_Static<double, 9>(0., J[0]); 
            J[0][0] = w;
            J[1][1] = w;
            J[2][2] = w;
            return;
        }

        double centroid[3];
        GetCentroid(k, centroid);

        const double w = _w / _s.size();
        makeInterpolatedVector_Static<double, 3>(w, _V.GetVertex(k), -w, centroid, J[0]);
    }

protected:
    const VertexNode & _V;
    const vector<const ScaleNode *> _s;
    const Mesh & _mesh;
    const double _w;

    vector<vector<int> *> _allOneRings;
};

#endif
