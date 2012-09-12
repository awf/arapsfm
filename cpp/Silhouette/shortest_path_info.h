#ifndef __SHORTEST_PATH_INFO_H__
#define __SHORTEST_PATH_INFO_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Geometry/mesh.h"

struct ShortestPathInfo
{
    ShortestPathInfo(const Mesh & Mesh_,
                     const Matrix<double> & SilCandDistances_,
                     const Matrix<int> & SilEdgeCands_,
                     const Vector<double> & SilEdgeCandParam_,
                     const Vector<int> & SilCandAssignedFaces_,
                     const Matrix<double> & SilCandU_)
        : Mesh(Mesh_),
          SilCandDistances(SilCandDistances_),
          SilEdgeCands(SilEdgeCands_),
          SilEdgeCandParam(SilEdgeCandParam_),
          SilCandAssignedFaces(SilCandAssignedFaces_),
          SilCandU(SilCandU_)
        {}

    const Mesh & Mesh;                          // mesh information is defined on
    const Matrix<double> & SilCandDistances;    // (precomputed) truncated geodesic distance matrix
    const Matrix<int> & SilEdgeCands;           // edges which defined preimage points
    const Vector<double> & SilEdgeCandParam;    // linear coordinates along edges which define preimage points
    const Vector<int> & SilCandAssignedFaces;   
    const Matrix<double> & SilCandU;            
};

#endif
