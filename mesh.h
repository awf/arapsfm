#ifndef __MESH_H__
#define __MESH_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include <utility>
#include <map>
#include <algorithm>
using namespace std;

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

    /*
    double GetCotanWeight(const Matrix<double> & V, int halfEdgeIndex) const
    {


    }
    */

protected:
    const int _numVertices;
    const Matrix<int> & _triangles;
    vector<vector<int>> _vertexToHalfEdges;
    Vector<int> _oppositeHalfEdge;
};

#endif

