#ifndef __STATIC_GEOMETRY_H__
#define __STATIC_GEOMETRY_H__

#include "Math/static_linear.h"
#include "Geometry/mesh.h"

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
