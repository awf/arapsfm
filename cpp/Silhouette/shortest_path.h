#ifndef __SHORTEST_PATH_H__
#define __SHORTEST_PATH_H__

#include "shortest_path_info.h"
#include <utility>
using namespace std;

// ShortestPathSolver
struct ShortestPathSolver
{
    struct ShortestPath_PQData
    {
        double E;
        int i;
        pair<int, int> startEndIndices;
        int pathIndex;
    };

    struct ShortestPath_ComparePQData
    {
        bool operator()(const ShortestPath_PQData & d1, const ShortestPath_PQData & d2)
        {
            if (d2.E < d1.E) 
                return true;
            else if (d2.E == d1.E)
                return d2.i < d1.i;
            
            return false;
        }
    };

    ShortestPathSolver(const Matrix<double> & S,          // silhouette points
                       const Matrix<double> & SN,         // silhouette normals
                       const ShortestPathInfo & info,     // required info about the mesh and state
                       const Vector<double> & lambdas,    // lambdas for projection and normal errors
                       bool verbose);

    int getPathLength(const pair<int, int> * bounds = nullptr) const;

    void updateCandidates(const Matrix<double> & V1);

    double solveShortestPath(const pair<int, int> & bounds,    // bounds over the silhouette indices (inclusive at start and end)
                             Vector<int> & shortestPath,       // vector which the path will be written into
                             const pair<int, int> * startIndices = nullptr,
                             const pair<int, int> * endIndices = nullptr,
                             bool isCircular = false) const;

    double solveCircularShortestPath(const pair<int, int> & bounds,
                                     Vector<int> & shortestPath) const;

    double solve(Vector<int> & shortestPath, bool isCircular = true) const;

    double solveOnMesh(Vector<int> & faceLookup, Matrix<double> & U, bool isCircular) const;

protected:
    const Matrix<double> & _S;
    const Matrix<double> & _SN;
    const ShortestPathInfo & _info;
    const int _numCandidates, _numEdgeCandidates, _numVertices;
    const Vector<double> & _lambdas;

    Matrix<double> _Q;
    Matrix<double> _QN;

    bool _verbose;
};

#endif
