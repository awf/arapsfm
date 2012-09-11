#ifndef __LAPLACIAN_H__
#define __LAPLACIAN_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/mesh.h"

// laplacianResiduals_Unsafe
inline void laplacianResiduals_Unsafe(const Mesh & mesh, const Matrix<double> & V, const int vertexIndex, const double & w, double * e)
{
    // get the indices for the (directional) incident edges
    const vector<int> & halfEdges = mesh.GetHalfEdgesFromVertex(vertexIndex);

    // calculate the centroid
    double centroid[3] = {0., 0., 0.};
    const double wSum = (double)halfEdges.size();

    for (int i=0; i < halfEdges.size(); i++)
    {
        const int j = mesh.GetHalfEdge(halfEdges[i], 1);
        makeInterpolatedVector_Static<double, 3>(1.0, centroid, 1.0, V[j], centroid);
    }

    // calulate the weighted error
    makeInterpolatedVector_Static<double, 3>(w, V[vertexIndex], -w / wSum, centroid, e);
}

// laplacianJac_Vi_Unsafe
inline void laplacianJac_Vi_Unsafe(const Mesh & mesh, const int vertexIndex, const int whichParam, const double & w, double * J)
{
    // get the indices for the (directional) incident edges
    const vector<int> & halfEdges = mesh.GetHalfEdgesFromVertex(vertexIndex);
    int indexIntoAdj = whichParam - 1;

    double diagonalEntry;

    if (indexIntoAdj < 0)
    {
        // centre vertex
        diagonalEntry = w;
    }
    else
    {
        const double wSum = (double)halfEdges.size();

        // adjacent vertex
        diagonalEntry = -w * (1.0 / wSum);
    }

    fillVector_Static<double, 9>(0., J); 
    J[0] = diagonalEntry;
    J[4] = diagonalEntry;
    J[8] = diagonalEntry;
}

// LaplacianEnergy
class LaplacianEnergy : public Energy
{
public:
    LaplacianEnergy(const VertexNode & V, const Mesh & mesh, const double w)
        : _V(V), _mesh(mesh), _w(w)
    {}

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
            vector<int> * pUsedParamTypes = new vector<int>(i->first, _V.GetParamId());
            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, i->second));
        }
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        return (*_allOneRings[k])[i] + _V.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const 
    {
        return _mesh.GetNumberOfVertices();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        laplacianResiduals_Unsafe(_mesh, _V.GetVertices(), k, _w, &e[0]);
    }
    
    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        laplacianJac_Vi_Unsafe(_mesh, k, whichParam, _w, J[0]);
    }

protected:
    const VertexNode & _V;
    const Mesh & _mesh;
    const double _w;

    vector<vector<int> *> _allOneRings;
};

#endif
