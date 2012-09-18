/* shortest_path.cpp */
#include "Silhouette/shortest_path.h"
#include "Math/static_linear.h"

#include <utility>
#include <limits>
#include <queue>
#include <list>
#include <cassert>
#include <iostream>
using namespace std;

ShortestPathSolver::ShortestPathSolver(const Matrix<double> & S,    // silhoutte points
                   const Matrix<double> & SN,                       // silhoutte normals
                   const ShortestPathInfo & info,                   // required info about the mesh and state
                   const Vector<double> & lambdas,                  // lambdas for projection and normal errors
                   bool verbose)
    : _S(S), _SN(SN), 
      _info(info),
      _numCandidates(info.SilCandDistances.num_rows()),
      _numEdgeCandidates(info.SilEdgeCands.num_rows()),
      _numVertices(_numCandidates - _numEdgeCandidates),
      _Q(_numCandidates, 3), _QN(_numCandidates, 3), 
      _lambdas(lambdas), 
      _verbose(verbose)
{
}

int ShortestPathSolver::getPathLength(const pair<int, int> * bounds) const
{
    // calculate the path length. NOTE that bounds.second is INCLUDED in the path
    if (bounds == nullptr)
        return _S.num_rows();

    int pathLength;

    if (bounds->second > bounds->first)
        pathLength = bounds->second - bounds->first + 1;
    else
        pathLength = bounds->second + _S.num_rows() - bounds->first + 1;

    return pathLength;
}

void ShortestPathSolver::updateCandidates(const Matrix<double> & V1)
{
    // candidate points

    // first candidate points are vertices
    copy(V1[0], V1[_numVertices], _Q[0]);

    // generate normalised vertex normals
    for (int i=0; i < _numVertices; i++)
    {
        double * n = _QN[i];
        vertexNormal_Unsafe(_info._Mesh, V1, i, n);
        normalizeVector_Static<double, 3>(n);
    }

    // next candidate points on edges and edge normals
    for (int l=0; l < _numEdgeCandidates; l++)
    {
        const int i = _info.SilEdgeCands[l][0], j = _info.SilEdgeCands[l][1];
        const double & t = _info.SilEdgeCandParam[l];

        // position
        makeInterpolatedVector_Static<double, 3>(1 - t, V1[i], t, V1[j], _Q[_numVertices + l]);

        // normal
        double * n = _QN[_numVertices + l];
        makeInterpolatedVector_Static<double, 3>(1 - t, _QN[i], t, _QN[j], n);
        normalizeVector_Static<double, 3>(n);
    }
}

double ShortestPathSolver::solveShortestPath(const pair<int, int> & bounds,    // bounds over the silhouette indices (inclusive at start and end)
                         Vector<int> & shortestPath,            // vector which the path will be written into
                         const pair<int, int> * startIndices,
                         const pair<int, int> * endIndices,
                         bool isCircular) const
{
    // calculate the path length. NOTE that bounds.second is INCLUDED in the path
    int pathLength = getPathLength(&bounds);

    // allocate the energy matrix and apply any constraints for the start indices
    Matrix<double> E(pathLength + 1, _Q.num_rows());
    fill(E[0], E[1], 0);

    if (startIndices != nullptr)
    {
        assert(startIndices->first < startIndices->second);
        fill(E[0], E[0] + startIndices->first, numeric_limits<double>::max());
        fill(E[0] + startIndices->second, E[1], numeric_limits<double>::max());

        if (_verbose)
        {
            cout << "ShortestPathSolver::solveShortestPath: (start, end) = (" << 
                startIndices->first << "," << startIndices->second << ")" << endl;
        }
    }

    if (endIndices != nullptr)
        assert(endIndices->first < endIndices->second);

    Matrix<int> previous(pathLength, _Q.num_rows());

    bool done = false;
    int currentIteration = 0;
    int i = bounds.first;

    while (!done)
    {
        if (isCircular || currentIteration > 0)
        {
            const double * energyBeforeJ = E[currentIteration];

            for (int j=0; j < _Q.num_rows(); j++)
            {
                // find minimum energy to j and the cheapest vertex to come from
                double minEnergyToJ = numeric_limits<double>::max();
                int minIndexToJ = -1;

                for (int l=0; l < _Q.num_rows(); l++)
                {
                    double energyToJ = _lambdas[0] * _info.SilCandDistances[j][l] + energyBeforeJ[l];

                    if (energyToJ < minEnergyToJ)
                    {
                        minEnergyToJ = energyToJ;
                        minIndexToJ = l;
                    }
                }

                // update energy and previous preimage vertex
                E[currentIteration + 1][j] = minEnergyToJ;
                previous[currentIteration][j] = minIndexToJ;
            }

        }
        else
        {
            copy(E[0], E[1], E[1]);
        }

        // add the projection and normal errors
        const double * Si = _S[i];
        const double * SNi = _SN[i];

        // force end constraints here
        bool forceEndConstraints = (i == bounds.second) && (endIndices != nullptr);

        for (int j=0; j < _Q.num_rows(); j++)
        {
            if (forceEndConstraints && !((endIndices->first <= j) && (j < endIndices->second)))
            {
                E[currentIteration + 1][j] = numeric_limits<double>::max();
            }
            else
            {
                // projection error
                double projEnergy = 0.;
                for (int k=0; k<2; k++)
                {
                    double projDiff = Si[k] - _Q[j][k];
                    projEnergy += projDiff*projDiff;
                }

                // normal error
                double normEnergy = _QN[j][2] * _QN[j][2];
                for (int k=0; k < 2; k++)
                {
                    double normDiff = SNi[k] - _QN[j][k];
                    normEnergy += normDiff*normDiff;
                }

                E[currentIteration + 1][j] += _lambdas[1] * projEnergy + _lambdas[2] * normEnergy;
            }

        }

        // terminate or continue?
        if ((currentIteration > 0) && (i == bounds.second))
        {
            done = true;
        }
        else
        {
            i = (i + 1) % _S.num_rows();
        }
        
        currentIteration++;
    }

    // construct the reverse path
    int l = 0;
    double minFinalEnergy = numeric_limits<double>::max();
    int minFinalIndex = -1;

    // find the minimum final vertex
    for (l = 0; l < E.num_cols(); l++)
    {
        if (E[currentIteration][l] < minFinalEnergy)
        {
            minFinalEnergy = E[currentIteration][l];
            minFinalIndex = l;
        }
    }

    int pathIndex = isCircular ? pathLength : pathLength - 1;
    shortestPath[pathIndex] = minFinalIndex;

    if (_verbose)
        cout << "ShortestPathSolver::solveShortestPath: minFinalEnergy = " << minFinalEnergy << endl;

    // proceed backwards to fill in the path
    pathIndex--;
    i = currentIteration - 1;
    l = minFinalIndex;

    while (pathIndex >= 0)
    {
        l = previous[i][l];
        shortestPath[pathIndex] = l;

        pathIndex--;
        i--;
    }

    return minFinalEnergy;
}

double ShortestPathSolver::solveCircularShortestPath(const pair<int, int> & bounds,
                                 Vector<int> & shortestPath) const
{
    // calculate the path length. NOTE that bounds.second is INCLUDED in the path
    int pathLength = getPathLength(&bounds);

    // get the number of candidate silhouette preimage positions
    int N = _Q.num_rows();

    // test open shortest path

    // newPath is size pathLength + 1 to store the last preimage candidate index
    // at the front AND back of the vector. This is required because the path is unwrapped and
    // for a non-circular path these entries are different.
    ShortestPath_PQData d;
    d.i = -1;
    d.pathIndex = 0;
    d.startEndIndices = pair<int, int>(0, N);

    vector<Vector<int> *> pathVectors;
    pathVectors.push_back(new Vector<int>(pathLength + 1));

    Vector<int> * path = pathVectors[d.pathIndex];
    
    d.E = solveShortestPath(bounds, *path, &d.startEndIndices, &d.startEndIndices, true);

    // initialise priority queue
    priority_queue<ShortestPath_PQData, vector<ShortestPath_PQData>, ShortestPath_ComparePQData> pq;
    pq.push(d);

    // initialise list of indices into pathVectors which can be reused
    list<int> freePathVectors;

    // initialise iteration counter
    int i = 0;

    while (!pq.empty())
    {
        // view top element on the priority queue
        const ShortestPath_PQData & t = pq.top();
        path = pathVectors[t.pathIndex];

        if (_verbose)
            cout << "ShortestPathSolver::solveCircularShortestPath: E = " << t.E << endl;

        // return if circular
        if ((*path)[pathLength] == (*path)[0])
        {
            cout << "[>] ShortestPathSolver::solveCircularShortestPath: E = " << t.E << endl;
            // unsafe copy of path. Skip the repeated preimage candidate index for the last silhoutte position
            copy(path->begin() + 1, path->end(), shortestPath.begin());

            for (int l=0; l < pathVectors.size(); l++)
                delete pathVectors[l];

            return t.E;
        }
        
        // no longer need pointer so put it onto `freePathVectors`
        freePathVectors.push_back(t.pathIndex);

        // split the lower bound path
        int pathStart = (*path)[0], pathEnd = (*path)[pathLength];

        if (pathStart > pathEnd)
        {
            int temp = pathStart; pathStart = pathEnd; pathEnd = temp;
        }

        int split = (pathStart + pathEnd) >> 1;
        int indices[3] = {pathStart, split, pathEnd};

        // done with the top element
        pq.pop();

        // add the next splits to the priority queue
        for (int j=0; j < 2; j++)
        {
            if (indices[j] >= indices[j+1]) continue;

            ShortestPath_PQData e;
            e.i = i++;
            e.startEndIndices = pair<int, int>(indices[j], indices[j+1]);

            if (!freePathVectors.empty())
            {
                e.pathIndex = freePathVectors.front();
                freePathVectors.pop_front();
            }
            else
            {
                e.pathIndex = pathVectors.size();
                pathVectors.push_back(new Vector<int>(pathLength + 1));
            }

            e.E = solveShortestPath(bounds, *pathVectors[e.pathIndex], &e.startEndIndices, &e.startEndIndices, true);
            pq.push(e);
        }
    }

    return numeric_limits<double>::max();
}

double ShortestPathSolver::solve(Vector<int> & shortestPath, bool isCircular) const
{
    // default bounds
    pair<int, int> bounds(0, _S.num_rows() - 1);

    if (isCircular)
        return solveCircularShortestPath(bounds, shortestPath);
    
    return solveShortestPath(bounds, shortestPath);
}

double ShortestPathSolver::solveOnMesh(Vector<int> & faceLookup, Matrix<double> & U, bool isCircular) const
{
    // default bounds
    pair<int, int> bounds(0, _S.num_rows() - 1);

    Vector<int> shortestPath(getPathLength());

    double E;

    if (isCircular)
        E = solveCircularShortestPath(bounds, shortestPath);
    else
        E = solveShortestPath(bounds, shortestPath);

    // translate shortest path to faces and barycentric coordinates
    for (int i = 0; i < shortestPath.size(); i++)
    {
        int preimageIndex = shortestPath[i];
        faceLookup[i] = _info.SilCandAssignedFaces[preimageIndex];
        copy(_info.SilCandU[preimageIndex], _info.SilCandU[preimageIndex + 1], U[i]);
    }

    return E;
}

