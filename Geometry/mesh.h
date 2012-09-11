#ifndef __MESH_H__
#define __MESH_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include <utility>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
using namespace std;

#include <cmath>
#include "Math/static_linear.h"
#include "Util/debug_messages.h"

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
                int vertexIndex = _triangles[i][j];

                _vertexToHalfEdges[vertexIndex].push_back(halfEdgeIndex);

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

    // Topological queries
    const int * GetTriangle(int triangleIndex) const
    {
        return _triangles[triangleIndex];
    }

    int GetNumberOfVertices() const { return _numVertices; }
    int GetNumberOfHalfEdges() const { return _oppositeHalfEdge.size(); }

    int GetHalfEdgeTriangle(int halfEdgeIndex) const
    {
        return halfEdgeIndex / 3;
    }

    int GetHalfEdgeOffset(int halfEdgeIndex) const
    {
        return halfEdgeIndex % 3;
    }

    int GetHalfEdge(int halfEdgeIndex, int whichVertex) const
    {
        int triangleIndex = GetHalfEdgeTriangle(halfEdgeIndex);
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

    const vector<int> & GetHalfEdgesFromVertex(int vertexIndex) const
    {
        return _vertexToHalfEdges[vertexIndex];
    }

    int GetAdjacentTriangle(int triangleIndex, int edgeIndex) const
    {
        int halfEdgeIndex = 3*triangleIndex + edgeIndex;
        int oppositeIndex = _oppositeHalfEdge[halfEdgeIndex];

        return oppositeIndex != -1 ? GetHalfEdgeTriangle(oppositeIndex) : -1;
    }

    vector<int> GetNRing(int vertexIndex, int N, bool includeSource=true) const
    {
        vector<int> nring;

        Vector<unsigned char> explored(_numVertices);
        fillVector(0, explored);

        priority_queue<pair<int, int>, vector<pair<int, int>>, NRingComparison> pq;
        pq.push(pair<int, int>(0, vertexIndex));

        bool pastFirst = false;

        while (pq.size() > 0)
        {
            // pop the top item
            auto next = pq.top();
            int depth = next.first;
            int nextVertex = next.second;
            pq.pop();

            // continue if already explored
            if (explored[nextVertex]) 
                continue;

            // no need to explore further
            if (depth > N)
                break;

            // add the vertex to the nring
            if (pastFirst || includeSource)
                nring.push_back(nextVertex);

            pastFirst = true;

            // add other half edges to the queue
            auto halfEdges = GetHalfEdgesFromVertex(nextVertex);
            for (auto i = halfEdges.begin(); i != halfEdges.end(); i++)
            {
                int adjVertex = GetHalfEdge((*i), 1);
                if (explored[adjVertex]) continue;

                pq.push(pair<int, int>(depth + 1, adjVertex));
            }

            // mark as explored
            explored[nextVertex] = 1;
        }

        return nring;
    }

    // Geometry queries
    double GetCotanWeight(const Matrix<double> & V, int halfEdgeIndex) const
    {
        int halfEdges [] = { halfEdgeIndex, GetOppositeHalfEdge(halfEdgeIndex) };

        double w = 0.;

        for (int q = 0; q < 2; q++)
        {
            halfEdgeIndex = halfEdges[q];
            if (halfEdgeIndex == -1)
                continue;

            int halfEdgeFace = GetHalfEdgeTriangle(halfEdgeIndex);
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
    // NRingComparison
    struct NRingComparison
    {
        bool operator()(const pair<int, int> & l, const pair<int, int> & r)
        {
            if (r.first < l.first)
                return true;
            else if (r.first == l.first)
            {
                return r.second < l.second;
            }

            return false;
        }
    };

    const int _numVertices;
    const Matrix<int> & _triangles;
    vector<vector<int>> _vertexToHalfEdges;
    Vector<int> _oppositeHalfEdge;
};

#endif

