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
            {
                _oppositeHalfEdge[l1] = -1;

                // mesh is open so store extra adjacent vertex in map
                auto j = _boundaryVertices.find(GetHalfEdge(l1, 1));
                if (j == _boundaryVertices.end())
                {
                    vector<int> vertexPairs(1, GetHalfEdge(l1, 0));
                    _boundaryVertices.insert(pair<int, vector<int>>(GetHalfEdge(l1, 1), vertexPairs));
                }
                else
                {
                    j->second.push_back(GetHalfEdge(l1, 0));
                }
            }
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

    vector<int> GetTrianglesAtVertex(int vertexIndex) const
    {
        auto halfEdges = GetHalfEdgesFromVertex(vertexIndex); 

        vector<int> triangles(halfEdges.size());
        for (int i = 0; i < halfEdges.size(); i++)
            triangles[i] = GetHalfEdgeTriangle(halfEdges[i]);

        return triangles;
    }

    vector<int> GetAdjacentVertices(int vertexIndex, bool includeSource = false) const
    {
        vector<int> adjVertices;
        if (includeSource) adjVertices.push_back(vertexIndex);

        auto halfEdges = GetHalfEdgesFromVertex(vertexIndex);

        for (auto i = halfEdges.begin(); i != halfEdges.end(); i++)
            adjVertices.push_back(GetHalfEdge(*i, 1));

        auto j = _boundaryVertices.find(vertexIndex);
        if (j != _boundaryVertices.end())
            copy(j->second.begin(), j->second.end(), back_inserter(adjVertices));

        return adjVertices;
    }

    vector<int> GetNRing(int vertexIndex, int N, bool includeSource = true) const
    {
        // special case
        if (N == 1) return GetAdjacentVertices(vertexIndex, includeSource);

        vector<int> nring;
        Vector<bool> explored(_numVertices);
        fillVector(false, explored);

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
            vector<int> adjVertices = GetAdjacentVertices(nextVertex);
            for (auto i = adjVertices.begin(); i != adjVertices.end(); i++)
            {
                int adjVertex = *i;
                if (explored[adjVertex]) continue;

                pq.push(pair<int, int>(depth + 1, adjVertex));
            }

            // mark as explored
            explored[nextVertex] = true;
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
    map<int, vector<int>> _boundaryVertices;
};

// Geometric functions

// faceNormal_Unsafe (UNNORMALISED)
inline void faceNormal_Unsafe(const double * a, const double * b, const double * c, double * n)
{
    double ac[3], bc[3];
    subtractVectors_Static<double, 3>(a, c, ac);
    subtractVectors_Static<double, 3>(b, c, bc);

    crossProduct_Static(ac, bc, n);
}

// faceNormalJac_Unsafe
inline void faceNormalJac_Unsafe(const double * a, const double * b, const double * c, double * J)
{
    J[0] = 0;
    J[1] = b[2] - c[2];
    J[2] = -b[1] + c[1];
    J[3] = 0;
    J[4] = -a[2] + c[2];
    J[5] = a[1] - c[1];
    J[6] = 0;
    J[7] = a[2] - b[2];
    J[8] = -a[1] + b[1];
    J[9] = -b[2] + c[2];
    J[10] = 0;
    J[11] = b[0] - c[0];
    J[12] = a[2] - c[2];
    J[13] = 0;
    J[14] = -a[0] + c[0];
    J[15] = -a[2] + b[2];
    J[16] = 0;
    J[17] = a[0] - b[0];
    J[18] = b[1] - c[1];
    J[19] = -b[0] + c[0];
    J[20] = 0;
    J[21] = -a[1] + c[1];
    J[22] = a[0] - c[0];
    J[23] = 0;
    J[24] = a[1] - b[1];
    J[25] = -a[0] + b[0];
    J[26] = 0;
}

// vertexNormal_Unsafe (UNNORMALISED)
inline void vertexNormal_Unsafe(const Mesh & mesh, const Matrix<double> & V1, int vertexId, double * n)
{
    fillVector_Static<double, 3>(0, n);

    vector<int> adjacentTriangles = mesh.GetTrianglesAtVertex(vertexId);

    for (int i=0; i < adjacentTriangles.size(); i++)
    {
        const int * Tj = mesh.GetTriangle(adjacentTriangles[i]);

        double faceNormal[3];
        faceNormal_Unsafe(V1[Tj[0]], V1[Tj[1]], V1[Tj[2]], faceNormal);

        addVectors_Static<double, 3>(faceNormal, n, n);
    }
}

// vertexNormalJac
inline Matrix<double> vertexNormalJac(const Mesh & mesh, const Matrix<double> & V1, int vertexId)
{
    // get the summed face normals jacobian
    vector<int> incOneRing = mesh.GetNRing(vertexId, 1, true);
    vector<int> adjTriangles = mesh.GetTrianglesAtVertex(vertexId);

    Matrix<double> sumFaceNormalsJac(3, 3*incOneRing.size(), 0.);

    // build an index mapping for `incOneRing` (which is ordered)
    std::map<int, int> indexIncOneRing;
    auto it = incOneRing.begin();
    for (int l=0; it != incOneRing.end(); l++, it++)
        indexIncOneRing.insert(std::pair<int, int>(*it, l));

    for (int i=0; i < adjTriangles.size(); i++)
    {
        // calculate the face normal Jacobian for the vertices in the face
        const int * Ti = mesh.GetTriangle(adjTriangles[i]);

        double faceNormalJac[27];
        faceNormalJac_Unsafe(V1[Ti[0]], V1[Ti[1]], V1[Ti[2]], faceNormalJac);
        
        // add the columns into the appropriate columns in sumFaceNormalsJac
        for (int j=0; j < 3; j++)
        {
            auto indexPairPointer = indexIncOneRing.find(Ti[j]);
            assert(indexPairPointer != indexIncOneRing.end());

            int columnIndex = indexPairPointer->second;

            for (int r=0; r < 3; r++)
                for (int c=0; c < 3; c++)
                    sumFaceNormalsJac[r][3*columnIndex + c] += faceNormalJac[9*r + 3*j + c];
        }
    }

    return sumFaceNormalsJac;
}

#endif

