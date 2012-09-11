#ifndef __TEST_MESH_H__
#define __TEST_MESH_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Util/pyarray_conversion.h"
#include "Geometry/mesh.h"

#include <iostream>
using namespace std;

int test_mesh(PyArrayObject * npy_vertices,
              PyArrayObject * npy_triangles)
{
    PYARRAY_AS_MATRIX(double, npy_vertices, vertices);
    PYARRAY_AS_MATRIX(int, npy_triangles, triangles);
    const int numVertices = vertices.num_rows();

    Mesh mesh(numVertices, triangles);

    cout << "NumberOfVertices: " << mesh.GetNumberOfVertices() << endl;
    cout << "NumberOfHalfEdges: " << mesh.GetNumberOfHalfEdges() << endl;
    cout << "All \"HalfEdge\"'s: " << endl;

    // GetOppositeHalfEdge
    for (int i=0; i < mesh.GetNumberOfHalfEdges(); i++)
    {
        cout << " " << i << ": (" << mesh.GetHalfEdge(i, 0) << "," << mesh.GetHalfEdge(i, 1) << ") <=> ";

        int j = mesh.GetOppositeHalfEdge(i);
        if (j == -1)
            cout << "None" << endl;
        else
            cout << "(" << mesh.GetHalfEdge(j, 0) << "," << mesh.GetHalfEdge(j, 1) << ")" << endl;
    }

    // GetHalfEdgesFromVertex
    cout << "Vertex to \"HalfEdge\"'s: " << endl;

    for (int i=0; i < mesh.GetNumberOfVertices(); i++)
    {
        auto halfEdges = mesh.GetHalfEdgesFromVertex(i);

        cout << " " << i << ": ";

        for (auto j = halfEdges.begin(); j != halfEdges.end(); j++)
            cout << *j << " ";

        cout << endl;
    }

    // GetCotanWeight
    cout << "\"HalfEdge\" weights: " << endl;

    for (int i=0; i < mesh.GetNumberOfHalfEdges(); i++)
    {
        cout << " " << i << ": " << mesh.GetCotanWeight(vertices, i) << endl;;
    }

    return 0;
}

PyArrayObject * get_nring(PyArrayObject * npy_vertices,
                          PyArrayObject * npy_triangles,
                          int vertex,
                          int N,
                          bool includeSource)
{
    PYARRAY_AS_MATRIX(double, npy_vertices, vertices);
    PYARRAY_AS_MATRIX(int, npy_triangles, triangles);
    const int numVertices = vertices.num_rows();

    Mesh mesh(numVertices, triangles);

    vector<int> nring = mesh.GetNRing(vertex, N, includeSource);

    npy_intp dim = nring.size();

    PyArrayObject * npy_retNRing = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_INT32);
    PYARRAY_AS_VECTOR(int, npy_retNRing, retNRing);

    for (int i=0; i < nring.size(); i++)
        retNRing[i] = nring[i];

    return npy_retNRing;
}

#endif
