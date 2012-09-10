#ifndef __TEST_MESH_H__
#define __TEST_MESH_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "pyarray_conversion.h"
#include "mesh.h"

#include <iostream>
using namespace std;

int test_mesh(int numVertices,
              PyArrayObject * npy_triangles)
{
    PYARRAY_AS_MATRIX(int, npy_triangles, triangles);

    Mesh mesh(numVertices, triangles);

    cout << "NumberOfVertices: " << mesh.GetNumberOfVertices() << endl;
    cout << "NumberOfHalfEdges: " << mesh.GetNumberOfHalfEdges() << endl;
    cout << "All \"HalfEdge\"'s: " << endl;

    for (int i=0; i < mesh.GetNumberOfHalfEdges(); i++)
    {
        cout << " " << i << ": (" << mesh.GetHalfEdge(i, 0) << "," << mesh.GetHalfEdge(i, 1) << ") <=> ";

        int j = mesh.GetOppositeHalfEdge(i);
        if (j == -1)
            cout << "None" << endl;
        else
            cout << "(" << mesh.GetHalfEdge(j, 0) << "," << mesh.GetHalfEdge(j, 1) << ")" << endl;
    }

    cout << "Vertex to \"HalfEdge\"'s: " << endl;

    for (int i=0; i < mesh.GetNumberOfVertices(); i++)
    {
        auto halfEdges = mesh.GetHalfEdgesFromVertex(i);

        cout << " " << i << ": ";

        for (auto j = halfEdges.begin(); j != halfEdges.end(); j++)
            cout << *j << " ";

        cout << endl;
    }

    return 0;
}

#endif
