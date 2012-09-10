#ifndef __MESH_H__
#define __MESH_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include <utility>
#include <map>
#include <algorithm>
using namespace std;

#include <cmath>
#include "static_linear.h"

// Mesh
class Mesh
{
public:
    Mesh(int numVertices, const Matrix<int> & triangles)
        : _numVertices(numVertices), 
          _triangles(triangles),
          _vertexToHalfEdges(numVertices),
          _oppositeHalfEdge(3*triangles.num_rows())
    {
        BuildAdjacencyInformation();
    }

    void BuildAdjacencyInformation()
    {
        map<pair<int, int>, vector<int>> fullEdgeToHalfEdges;

        for (int i=0; i < _triangles.num_rows(); i++)
            for (int j=0; j < 3; j++)
            {
                int halfEdgeIndex = 3*i + j;
                int vertexId = _triangles[i][j];

                _vertexToHalfEdges[vertexId].push_back(halfEdgeIndex);

                int v0 = GetHalfEdge(halfEdgeIndex, 0);
                int v1 = GetHalfEdge(halfEdgeIndex, 1);
                if (v0 < v1) swap(v0, v1);
                pair<int, int> fullEdge(v0, v1);

                auto k = fullEdgeToHalfEdges.find(fullEdge);
                if (k == fullEdgeToHalfEdges.end())
                {
                    vector<int> halfEdges(1, halfEdgeIndex);
                    fullEdgeToHalfEdges.insert(pair<pair<int, int>, vector<int>>(fullEdge, halfEdges));
                }
                else
                {
                    k->second.push_back(halfEdgeIndex);
                }
            }

        for (auto i = fullEdgeToHalfEdges.begin(); i != fullEdgeToHalfEdges.end(); i++)
        {
            const vector<int> & halfEdges = i->second;
            const int l1 = halfEdges[0];

            if (halfEdges.size() == 1)
                _oppositeHalfEdge[l1] = -1;
            else
            {
                const int l2 = halfEdges[1];
                _oppositeHalfEdge[l1] = l2;
                _oppositeHalfEdge[l2] = l1;
            }
        }
    }

    int GetNumberOfVertices() const { return _numVertices; }
    int GetNumberOfHalfEdges() const { return _oppositeHalfEdge.size(); }

    int GetHalfEdgeFace(int halfEdgeIndex) const
    {
        return halfEdgeIndex / 3;
    }

    int GetHalfEdgeOffset(int halfEdgeIndex) const
    {
        return halfEdgeIndex % 3;
    }

    int GetHalfEdge(int halfEdgeIndex, int whichVertex) const
    {
        int triangleIndex = GetHalfEdgeFace(halfEdgeIndex);
        int offset = GetHalfEdgeOffset(halfEdgeIndex);

        if (whichVertex == 0)
            return _triangles[triangleIndex][offset];
        else
            return _triangles[triangleIndex][(offset + 1) % 3];
    }

    int GetOppositeHalfEdge(int halfEdgeIndex) const
    {
        return _oppositeHalfEdge[halfEdgeIndex];
    }

    const vector<int> & GetHalfEdgesFromVertex(int vertexId) const
    {
        return _vertexToHalfEdges[vertexId];
    }

    double GetCotanWeight(const Matrix<double> & V, int halfEdgeIndex) const
    {
        int halfEdges [] = { halfEdgeIndex, GetOppositeHalfEdge(halfEdgeIndex) };

        double w = 0.;

        for (int q = 0; q < 2; q++)
        {
            halfEdgeIndex = halfEdges[q];
            if (halfEdgeIndex == -1)
                continue;

            int halfEdgeFace = GetHalfEdgeFace(halfEdgeIndex);
            int oppositeOffset = (GetHalfEdgeOffset(halfEdgeIndex) + 2) % 3;

            double l[3];

            for (int i=0; i < 3; i++)
            {
                const int m = _triangles[halfEdgeFace][(oppositeOffset + i) % 3];
                const int n = _triangles[halfEdgeFace][(oppositeOffset + i + 1) % 3];

                double d[3];
                subtractVectors_Static<double, 3>(V[m], V[n], d);

                l[i] = norm_L2_Static<double, 3>(d);
            }
            
            double t = (l[0]*l[0] + l[2]*l[2] - l[1]*l[1]) / (2*l[0]*l[2]);
            t = t > 1.0 ? 1.0 : t;
            t = t < -1.0 ? -1.0 : t;

            const double alpha = acos(t);
            const double addToW = 0.5 / tan(alpha);

            if (addToW < 1e-6)
                continue; 

            w += addToW;
        }

        return w;
    }

protected:
    const int _numVertices;
    const Matrix<int> & _triangles;
    vector<vector<int>> _vertexToHalfEdges;
    Vector<int> _oppositeHalfEdge;
};

#endif

