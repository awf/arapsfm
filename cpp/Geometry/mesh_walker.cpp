/* mesh_walker.cpp */

// includes
#include "Math/v3d_linear.h"
#include "Math/v3d_nonlinlsq.h"
using namespace V3D;

#include "Math/static_linear.h"  // Unsafe vector and matrix manipulation
#include "Geometry/quaternion.h"

#include <cmath>
#include <algorithm>
#include <iostream>

#include "Geometry/mesh_walker.h"
#include "Geometry/mesh.h"
#include "Util/debug_messages.h"

// definitions
#define BARYCENTRIC_EPS (1e-4)
#define LENGTH_THRESHOLD (1e-4)

MeshWalker::MeshWalker(const Mesh & mesh,
                       const Matrix<double> & V)
        : _mesh(mesh), _V(V)
{}

bool MeshWalker::applyDisplacement(Matrix<double> & U,
                                   Vector<int> & L,
                                   const VectorArrayAdapter<double> & delta) const
{
    bool aFaceHasChanged = false;

    for (int i=0; i < U.num_rows(); i++)
    {
        const int newFace = applySingleDisplacement(U[i], L[i], delta[i].begin());

        if (newFace != L[i])
            aFaceHasChanged = true;

        L[i] = newFace;
    }

    return aFaceHasChanged;
}

int MeshWalker::applySingleDisplacement(double * u_, int currentFace, const double * delta_) const
{
    double u[3] = {u_[0], u_[1], 1.0 - u_[0] - u_[1]};
    double delta[3] = {delta_[0], delta_[1], - delta_[0] - delta_[1]};

    int currentIteration = 0;

    while(true)
    {
        PRINT_VARIABLE(currentFace);
        PRINT_VECTOR3(u);
        PRINT_VECTOR3(delta);

        double nextU[3];
        int exitIndex = whichEdgeBroke(u, delta, nextU);

        PRINT_VARIABLE(exitIndex);
        PRINT_VECTOR3(nextU);

        // inside the triangle?
        if (exitIndex == -1)
        {
            std::copy(nextU, nextU+2, u_);
            return currentFace;
        }

        // at a manifold edge?
        int nextFace = _mesh.GetAdjacentTriangle(currentFace, exitIndex);
        PRINT_VARIABLE(nextFace);

        if (nextFace == -1)
        {
            std::copy(nextU, nextU+2, u_);
            return currentFace;
        }

        // determine the desired length in world coordinates and the heading
        double worldDelta[3];
        baryToPosition(currentFace, delta, worldDelta);
        double worldDeltaLength = norm_L2_Static<double, 3>(worldDelta);
        PRINT_VECTOR3(worldDelta);

        if (worldDeltaLength <= LENGTH_THRESHOLD)
        {
            std::copy(nextU, nextU+2, u_);
            return currentFace;
        }

        double heading[3];
        scaleVector_Static<double, 3>(1.0 / worldDeltaLength, worldDelta, heading);

        // determine original and reached position
        double originalPosition[3];
        baryToPosition(currentFace, u, originalPosition);
        PRINT_VECTOR3(originalPosition);

        double reachedPosition[3];
        baryToPosition(currentFace, nextU, reachedPosition);
        PRINT_VECTOR3(reachedPosition);

        // determine the amount traveled
        double traveled[3]; 
        subtractVectors_Static<double, 3>(reachedPosition, originalPosition, traveled);
        PRINT_VECTOR3(traveled);

        double traveledLength = norm_L2_Static<double, 3>(traveled);
        double remainingLength = worldDeltaLength - traveledLength;
        PRINT_VARIABLE(traveledLength);
        PRINT_VARIABLE(remainingLength);

        if (remainingLength <= LENGTH_THRESHOLD)
        {
            std::copy(nextU, nextU+2, u_);
            return currentFace;
        }

        // determine the edge that is entered in the next face
        const int * Ti = _mesh.GetTriangle(currentFace);
        int leavingEdge[2] = { Ti[exitIndex], Ti[(exitIndex + 1) % 3] };
        PRINT_VECTOR2(leavingEdge);

        const int * Tj = _mesh.GetTriangle(nextFace);
        double uNextFace[3];
        int entryIndex;

        for (entryIndex = 0; entryIndex < 3; entryIndex++)
        {
            int entryEdge[2] = { Tj[entryIndex], Tj[(entryIndex + 1) % 3] };

            if ((leavingEdge[0] == entryEdge[1]) && (leavingEdge[0] == entryEdge[1]))
            {
                uNextFace[(entryIndex + 1) % 3] = nextU[exitIndex];
                uNextFace[entryIndex] = nextU[(exitIndex + 1) % 3];
                uNextFace[(entryIndex + 2) % 3] = nextU[(exitIndex + 2) % 3];
                break;
            }
        }

        // determine the angle the heading makes with the leaving edge
        double leavingEdgeVector[3];
        subtractVectors_Static<double, 3>(_V[leavingEdge[1]], _V[leavingEdge[0]], leavingEdgeVector);
        normalizeVector_Static<double, 3>(leavingEdgeVector);
        PRINT_VECTOR3(leavingEdgeVector);

        double headingProj = innerProduct_Static<double, 3>(heading, leavingEdgeVector);
        headingProj = std::min(1.0, headingProj);
        headingProj = std::max(-1.0, headingProj);
        double angle = acos(headingProj);

        PRINT_VECTOR3(heading);
        PRINT_VARIABLE(angle);

        // determine the plane normal in the next face
        double planeNormal[3];
        calcPlaneNormal(nextFace, planeNormal);
        PRINT_VECTOR3(planeNormal);

        // rotate the edge vector by -angle
        double rotationVector[3];
        scaleVector_Static<double, 3>(-angle, planeNormal, rotationVector);
        double quatVector[4];
        quat_Unsafe(rotationVector, quatVector);
        double R[9];
        rotationMatrix_Unsafe(quatVector, R);

        double newHeading[3];
        multiply_A_v_Static<double, 3, 3>(R, leavingEdgeVector, newHeading);
        PRINT_VECTOR3(newHeading);

        // scale by the remaining length of the world delta
        scaleVectorIP_Static<double, 3>(remainingLength, newHeading);
        PRINT_VARIABLE(remainingLength);
        PRINT_VECTOR3(newHeading);

        // determine the desired world position
        double worldPosition[3];
        addVectors_Static<double, 3>(reachedPosition, newHeading, worldPosition);
        PRINT_VECTOR3(worldPosition);

        // resolve the new world position as an offset in barycentric coordinates in the next face
        double uWorldInNextFace[3];
        positionToBary(nextFace, worldPosition, uWorldInNextFace);
        PRINT_VECTOR3(uWorldInNextFace);

        subtractVectors_Static<double, 3>(uWorldInNextFace, uNextFace, delta);
        currentFace = nextFace;
        std::copy(uNextFace, uNextFace + 3, u);
    }
}


int MeshWalker::whichEdgeBroke(const double * u, const double * delta,
                               double * nextU) const
{
    // find limit 
    double t_ = 1.0;

    for (int i=0; i<3; i++)
    {
        if (delta[i] < 0.)
        {
            // check for 0.0
            double t = -u[i] / delta[i];
            if (t < t_) t_ = t;
        }
        else if (delta[i] > 0.)
        {
            // check for 1.0
            double t = (1.0 - u[i]) / delta[i];
            if (t < t_) t_ = t;
        }
        else
            continue;
    }

    // no edges hit?
    if (t_ >= 1.0) 
    {
        if (nextU != nullptr)
            addVectors_Static<double, 3>(u, delta, nextU);

        return -1;
    }

    double u1[3];
    makeInterpolatedVector_Static<double, 3>(1.0, u, t_, delta, u1);

    // massage into valid interior barycentric coordinates
    for (int i=0; i<3; i++)
    {
        if (u1[i] < BARYCENTRIC_EPS) u1[i] = 0.;
        if (u1[i] >= (1.0 - BARYCENTRIC_EPS)) u1[i] = 1.;
    }

    double sumU1 = sumVector_Static<double, 3>(u1);
    scaleVectorIP_Static<double, 3>(1.0 / sumU1, u1);

    // save if available
    if (nextU != nullptr)
        std::copy(u1, u1+3, nextU);

    // number of non-zero entries in the barycentric coordinates
    int nzCount = 0, nzIndices[3], zeroCount = 0, zeroIndices[3];
    for (int i=0; i<3; i++)
        if (u1[i] > 0.)
            nzIndices[nzCount++] = i;
        else
            zeroIndices[zeroCount++] = i;

    // at a vertex?
    if (nzCount == 1)
    {
        int i = nzIndices[0];

        if (delta[(i + 1) % 3] < delta[(i + 2) % 3])
            return (i + 2) % 3;
        else
            return i;
    }

    // at an edge?
    if (nzCount == 2)
        return (zeroIndices[0] + 1) % 3;

    assert(false);

    return -2;
}

void MeshWalker::baryToPosition(int faceIndex, const double * u, double * p) const
{
    const int * Ti = _mesh.GetTriangle(faceIndex);

    double partial[3];
    makeInterpolatedVector_Static<double, 3>(u[0], _V[Ti[0]], u[1], _V[Ti[1]], partial);
    makeInterpolatedVector_Static<double, 3>(1.0, partial, u[2], _V[Ti[2]], p);
}

void MeshWalker::positionToBary(int faceIndex, const double * p, double * u) const
{
    const int * Ti = _mesh.GetTriangle(faceIndex);
    double Vik[3], Vjk[3];

    double n[3];
    subtractVectors_Static<double, 3>(_V[Ti[0]], _V[Ti[2]], Vik);
    subtractVectors_Static<double, 3>(_V[Ti[1]], _V[Ti[2]], Vjk);
    crossProduct_Static<double>(Vik, Vjk, n);
    double sqrL2_n = sqrNorm_L2_Static<double, 3>(n);

    double Vji[3], pi[3], na[3];
    subtractVectors_Static<double, 3>(_V[Ti[1]], _V[Ti[0]], Vji);
    subtractVectors_Static<double, 3>(p, _V[Ti[0]], pi);
    crossProduct_Static<double>(Vji, pi, na);

    double Vkj[3], pj[3], nb[3];
    subtractVectors_Static<double, 3>(_V[Ti[2]], _V[Ti[1]], Vkj);
    subtractVectors_Static<double, 3>(p, _V[Ti[1]], pj);
    crossProduct_Static<double>(Vkj, pj, nb);

    u[0] = innerProduct_Static<double, 3>(n, nb) / sqrL2_n;
    u[2] = innerProduct_Static<double, 3>(n, na) / sqrL2_n;
    u[1] = 1.0 - u[0] - u[2];
}

void MeshWalker::calcPlaneNormal(int faceIndex, double * n) const
{
    const int * Ti = _mesh.GetTriangle(faceIndex);
    double Vik[3], Vjk[3];

    subtractVectors_Static<double, 3>(_V[Ti[0]], _V[Ti[2]], Vik);
    subtractVectors_Static<double, 3>(_V[Ti[1]], _V[Ti[2]], Vjk);

    crossProduct_Static(Vik, Vjk, n);
    normalizeVector_Static<double, 3>(n);
}


