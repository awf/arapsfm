#ifndef __MESH_WALKER_H__
#define __MESH_WALKER_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include <utility>
using namespace std;

#include "Geometry/mesh.h"

struct MeshWalker
{
    MeshWalker(const Mesh & mesh,
               const Matrix<double> & V);

    bool applyDisplacement(Matrix<double> & U,
                           Vector<int> & L,
                           const VectorArrayAdapter<double> & delta) const;
               
    int applySingleDisplacement(double * u_, int currentFace, const double * delta_) const;
    int whichEdgeBroke(const double * u, const double * delta, double * t = nullptr) const;

    const Matrix<double> & getVertices() const { return _V; }
    const Mesh & getMesh() const { return _mesh; }

protected:
    void baryToPosition(int faceIndex, const double * u, double * p) const;
    void positionToBary(int faceIndex, const double * p, double * u) const;
    void calcPlaneNormal(int faceIndex, double * n) const;

    const Mesh & _mesh;
    const Matrix<double> & _V;
};


#endif
