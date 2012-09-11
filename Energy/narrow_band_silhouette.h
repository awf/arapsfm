#ifndef __NARROW_BAND_SILHOUETTE_H__
#define __NARROW_BAND_SILHOUETTE_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/mesh.h"

#include <map>
#include <algorithm>
#include <set>
using namespace std;

// silhouetteProjResiduals_Unsafe
inline void silhouetteProjResiduals_Unsafe(const double * V1i, const double * V1j, const double * V1k, const double * u, const double * S, const double & w, double * e)
{
    e[0] = w*(S[0] - (u[0]*(V1i[0] - V1k[0]) + u[1]*(V1j[0] - V1k[0]) + V1k[0]));
    e[1] = w*(S[1] - (u[0]*(V1i[1] - V1k[1]) + u[1]*(V1j[1] - V1k[1]) + V1k[1]));
}

// silhouetteProjJac_V1l_Unsafe
inline void silhouetteProjJac_V1l_Unsafe(const double & u, const double & w, double * J)
{
    J[0] = -w*u;
    J[1] = 0;
    J[2] = 0;
    J[3] = 0;
    J[4] = -w*u;
    J[5] = 0;
}

// silhouetteProjJac_u_Unsafe
inline void silhouetteProjJac_u_Unsafe(const double * V1i, const double * V1j, const double * V1k, const double & w, double * J)
{
    J[0] = -w*(V1i[0] - V1k[0]);
    J[1] = -w*(V1j[0] - V1k[0]);
    J[2] = -w*(V1i[1] - V1k[1]);
    J[3] = -w*(V1j[1] - V1k[1]);
}

// SilhouetteProjectionEnergy
class SilhouetteProjectionEnergy : public Energy
{
public:
    SilhouetteProjectionEnergy(const VertexNode & V, const BarycentricNode & U, 
                        const Matrix<double> & S, const Mesh & mesh, 
                        const double w, const int narrowBand)
        : _V(V), _U(U), _S(S), _mesh(mesh), _w(w), _narrowBand(narrowBand)
    {}

    virtual ~SilhouetteProjectionEnergy()
    {
        for (int i = 0; i < _allNarrowBands.size(); i++)
            if (_allNarrowBands[i] != nullptr) delete _allNarrowBands[i];
    }

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) 
    {
        const Vector<int> & L = _U.GetFaceIndices();

        // mapping from n -> [k0, k1, ...]
        map<int, vector<int> *> narrowBandSizeToResidual;
        
        for (int i=0; i < L.size(); i++)
        {
            // source for the narrow band is taken as the first vertex in each face
            int sourceVertex = _mesh.GetTriangle(L[i])[0];

            // save the narrow band
            vector<int> * narrowBand = new vector<int>(_mesh.GetNRing(sourceVertex, _narrowBand, true));
            _allNarrowBands.push_back(narrowBand);

            // save the mapping from size to residual index
            int n = narrowBand->size();

            auto j = narrowBandSizeToResidual.find(n);
            if (j == narrowBandSizeToResidual.end())
            {
                vector<int> * residualMap = new vector<int>(1, i);
                narrowBandSizeToResidual.insert(pair<int, vector<int> *>(n, residualMap));
            }
            else
            {
                j->second->push_back(i);
            }
        }

        // construct cost functions
        for (auto i = narrowBandSizeToResidual.begin(); i != narrowBandSizeToResidual.end(); i++)
        {
            int n = i->first;
            vector<int> * pUsedParamTypes = new vector<int>(n + 1, _V.GetParamId());
            (*pUsedParamTypes)[0] = _U.GetParamId();

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 2, i->second));
        }
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        if (i == 0)
            return k + _U.GetOffset();

        return (*_allNarrowBands[k])[i-1] + _V.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        assert(false);
        return -1;
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        int faceIndex = _U.GetFaceIndex(k);
        const int * Ti = _mesh.GetTriangle(faceIndex);

        silhouetteProjResiduals_Unsafe(_V.GetVertex(Ti[0]), 
                                       _V.GetVertex(Ti[1]),
                                       _V.GetVertex(Ti[2]),
                                       _U.GetBarycentriCoordinate(k),
                                       _S[k],
                                       _w, 
                                       &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        const int * Ti = _mesh.GetTriangle(_U.GetFaceIndex(k));

        if (whichParam == 0)
        {
            // barycentric coordinate
            silhouetteProjJac_u_Unsafe(_V.GetVertex(Ti[0]), 
                                       _V.GetVertex(Ti[1]),
                                       _V.GetVertex(Ti[2]),
                                       _w, J[0]);
        }
        else
        {
            // vertex
            int j = (*_allNarrowBands[k])[whichParam - 1];

            // if face vertex
            if (j == Ti[0])
            {
                double u = _U.GetBarycentriCoordinate(k)[0];
                silhouetteProjJac_V1l_Unsafe(u, _w, J[0]);
            }
            else if (j == Ti[1])
            {
                double u = _U.GetBarycentriCoordinate(k)[1];
                silhouetteProjJac_V1l_Unsafe(u, _w, J[0]);
            }
            else if (j == Ti[2])
            {
                double u = 1.0 - _U.GetBarycentriCoordinate(k)[0] - _U.GetBarycentriCoordinate(k)[1];
                silhouetteProjJac_V1l_Unsafe(u, _w, J[0]);
            }
            else
            {
                // other vertex in the narrow band
                fillMatrix(J, 0);
            }
        }
    }

    virtual bool CanBeginIteration() const
    {
        // check that vertices of each face are in the narrow bands
        const Vector<int> & L = _U.GetFaceIndices();

        for (int i=0; i < L.size(); i++)
        {
            const int * Ti = _mesh.GetTriangle(L[i]);
            const vector<int> * narrowBand = _allNarrowBands[i];

            for (int j=0; j<3; j++)
            {
                auto l = find(narrowBand->begin(), narrowBand->end(), Ti[j]);

                if (l == narrowBand->end())
                {
                    // face vertex is not in the narrow band
                    return false;
                }
            }
        }

        return true;
    }

protected:
    const VertexNode & _V;
    const BarycentricNode & _U;
    const Matrix<double> & _S;
    const Mesh & _mesh;
    const double _w;
    const int _narrowBand;

    vector<vector<int> *> _allNarrowBands;
};

// Silhouette normal residual and derivatives (UNSAFE)

// silhouetteNormalResiduals_Unsafe
inline void silhouetteNormalResiduals_Unsafe(const Mesh & mesh, const Matrix<double> & V1, int faceIndex, const double * u_, const double * SN, const double & w, double * e)
{
    const int * Ti = mesh.GetTriangle(faceIndex);
    double vertexNormals[3][3];

    for (int i=0; i < 3; i++)
    {
        vertexNormal_Unsafe(mesh, V1, Ti[i], vertexNormals[i]);
        normalizeVector_Static<double, 3>(vertexNormals[i]);
    }

    double u[3] = { u_[0], u_[1], 1.0 - u_[0] - u_[1] };

    double normal[3];
    makeTriInterpolatedVector_Static<double, 3>(u, vertexNormals[0], vertexNormals[1], vertexNormals[2], normal);
    normalizeVector_Static<double, 3>(normal);

    e[0] = w*(SN[0] - normal[0]);
    e[1] = w*(SN[1] - normal[1]);
    e[2] = w*(- normal[2]);
}

// normalizationJac_Unsafe
inline void normalizationJac_Unsafe(const double * v, double * J)
{
    double x[10];

    x[0] = v[0]*v[0];
    x[1] = v[1]*v[1];
    x[2] = v[2]*v[2];
    x[3] = x[0] + x[1] + x[2];
    x[5] = 1.0 / sqrt(x[3]);
    x[4] = x[5]*x[5]*x[5];
    //x[4] = x[3]**(-3/2);
    //x[5] = x[4]**(1/3);
    x[6] = v[2]*x[4];
    x[7] = v[0]*v[1]*x[4];
    x[8] = v[0]*x[6];
    x[9] = v[1]*x[6];

    J[0] = -x[0]*x[4] + x[5];
    J[1] = -x[7];
    J[2] = -x[8];
    J[3] = -x[7];
    J[4] = -x[1]*x[4] + x[5];
    J[5] = -x[9];
    J[6] = -x[8];
    J[7] = -x[9];
    J[8] = -x[2]*x[4] + x[5];
}

// silhouetteNormalResidualsJac_V1
void silhouetteNormalResidualsJac_V1_Unsafe(const Mesh & mesh, const Matrix<double> & V1, int faceIndex, const double * u_, int vertexIndex, const double & w, double * J)
{
    // current triangle and full barycentric coordinates
    const int * Ti = mesh.GetTriangle(faceIndex);
    double u[3] = { u_[0], u_[1], 1.0 - u_[0] - u_[1] };

    // get one rings for all vertices in the face
    std::vector<int> oneRings[3];
    for (int i=0; i<3; i++)
        oneRings[i] = mesh.GetNRing(Ti[i], 1, true);

    // construct set of all vertices involved in the normal calculation
    std::set<int> allVertices;
    for (int i=0; i<3; i++)
        allVertices.insert(oneRings[i].begin(), oneRings[i].end());

    // get the index of the given vertex 
    auto j = allVertices.find(vertexIndex);
    if (j == allVertices.end())
    {
        // vertex index is not in the extended one ring so has no effect
        fillVector_Static<double, 9>(0., J);
        return;
    }
    int desiredIndex = *j;

    // build an index mapping for `allVertices` (which is ordered)
    std::map<int, int> indexAllVertices;
    auto it = allVertices.begin();
    for (int l=0; it != allVertices.end(); l++, it++)
        indexAllVertices.insert(std::pair<int, int>(*it, l));

    // get the normalised vertex normals and the independent summations to the final Jacobian
    double normalisedVertexNormals[3][3];
    Matrix<double> unJ(3, 3*allVertices.size(), 0.);

    for (int i=0; i<3; i++)
    {
        // get the vertex normal and normalisation jacobian
        double * vertexNormal = normalisedVertexNormals[i];
        vertexNormal_Unsafe(mesh, V1, Ti[i], vertexNormal);

        Matrix<double> normalizationJac(3,3);
        normalizationJac_Unsafe(vertexNormal, normalizationJac[0]);

        // normalise the vertex normal
        normalizeVector_Static<double, 3>(vertexNormal);

        // get the Jacobian for the given vertex (column ordering same as `oneRings[i]`)
        Matrix<double> unVertexNormalJ = vertexNormalJac(mesh, V1, Ti[i]);

        // apply the normalisation Jacobian (chain rule)
        Matrix<double> vertexJ(3, unVertexNormalJ.num_cols());
        multiply_A_B(normalizationJac, unVertexNormalJ, vertexJ);

        // add and scale the columns into the main J
        auto it = oneRings[i].begin();
        for (int l=0; it != oneRings[i].end(); it++, l++)
        {
            auto p = indexAllVertices.find(*it);
            assert(p != indexAllVertices.end());
            int m = p->second;

            for (int r=0; r<3; r++)
                for (int k=0; k<3; k++)
                    unJ[r][3*m + k] += u[i] * vertexJ[r][3*l + k];
        }
    }

    // construct blended (unnormalised) normal vector
    double normal[3];
    makeTriInterpolatedVector_Static<double, 3>(u, 
            normalisedVertexNormals[0],  
            normalisedVertexNormals[1], 
            normalisedVertexNormals[2], normal);

    // construct the final normalisation jacobian
    Matrix<double> normalizationJac(3,3);
    normalizationJac_Unsafe(normal, normalizationJac[0]);

    // apply to the final Jacobian (chain rule)
    Matrix<double> finalJ(3, unJ.num_cols());
    multiply_A_B(normalizationJac, unJ, finalJ);
    scaleMatrixIP(-w, finalJ);

    // return slice for the vertex of interest
    Matrix<double> outJ(3, 3, J);
    copyMatrixSlice(outJ, 0, 3*desiredIndex, 3, 3, outJ, 0, 0);
}

// silhouetteNormalResidualsJac_u_Unsafe
inline void silhouetteNormalResidualsJac_u_Unsafe(const Mesh & mesh, const Matrix<double> & V1, int faceIndex, const double * u_, const double & w, double * J)
{
    const int * Ti = mesh.GetTriangle(faceIndex);

    // get the normalised vertex normals
    double vertexNormals[3][3];
    for (int i=0; i < 3; i++)
    {
        double * vertexNormal = vertexNormals[i];
        vertexNormal_Unsafe(mesh, V1, Ti[i], vertexNormal);
        normalizeVector_Static<double, 3>(vertexNormal);
    }

    double unJ[3][2];
    for (int i=0; i < 3; i++)
        for (int j=0; j < 2; j++)
            unJ[i][j] = vertexNormals[j][i] - vertexNormals[2][i];

    // get the blended normal
    double normal[3];
    double u[3] = { u_[0], u_[1], 1.0 - u_[0] - u_[1] };

    makeTriInterpolatedVector_Static<double, 3>(u, 
            vertexNormals[0],  
            vertexNormals[1], 
            vertexNormals[2], normal);

    // get the normalisation Jacobian
    double normalizationJac[9];
    normalizationJac_Unsafe(normal, normalizationJac);

    // get the final jacobian
    multiply_A_B_Static<double, 3, 3, 2>(normalizationJac, unJ[0], J);
    scaleVectorIP_Static<double, 6>(-w, J);
}

// SilhouetteNormalEnergy
class SilhouetteNormalEnergy : public Energy
{
public:
    SilhouetteNormalEnergy(const VertexNode & V, const BarycentricNode & U, 
                        const Matrix<double> & SN, const Mesh & mesh, 
                        const double w, const int narrowBand)
        : _V(V), _U(U), _SN(SN), _mesh(mesh), _w(w), _narrowBand(narrowBand)
    {}

    virtual ~SilhouetteNormalEnergy()
    {
        for (int i = 0; i < _allNarrowBands.size(); i++)
            if (_allNarrowBands[i] != nullptr) delete _allNarrowBands[i];
    }

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) 
    {
        const Vector<int> & L = _U.GetFaceIndices();

        // mapping from n -> [k0, k1, ...]
        map<int, vector<int> *> narrowBandSizeToResidual;
        
        for (int i=0; i < L.size(); i++)
        {
            // source for the narrow band is taken as the first vertex in each face
            int sourceVertex = _mesh.GetTriangle(L[i])[0];

            // save the narrow band
            vector<int> * narrowBand = new vector<int>(_mesh.GetNRing(sourceVertex, _narrowBand, true));
            _allNarrowBands.push_back(narrowBand);

            // save the mapping from size to residual index
            int n = narrowBand->size();

            auto j = narrowBandSizeToResidual.find(n);
            if (j == narrowBandSizeToResidual.end())
            {
                vector<int> * residualMap = new vector<int>(1, i);
                narrowBandSizeToResidual.insert(pair<int, vector<int> *>(n, residualMap));
            }
            else
            {
                j->second->push_back(i);
            }
        }

        // construct cost functions
        for (auto i = narrowBandSizeToResidual.begin(); i != narrowBandSizeToResidual.end(); i++)
        {
            int n = i->first;
            vector<int> * pUsedParamTypes = new vector<int>(n + 1, _V.GetParamId());
            (*pUsedParamTypes)[0] = _U.GetParamId();

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 3, i->second));
        }
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        if (i == 0)
            return k + _U.GetOffset();

        return (*_allNarrowBands[k])[i-1] + _V.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        assert(false);
        return -1;
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        silhouetteNormalResiduals_Unsafe(_mesh, 
                                         _V.GetVertices(),
                                         _U.GetFaceIndex(k),
                                         _U.GetBarycentriCoordinate(k),
                                         _SN[k],
                                         _w,
                                         &e[0]);
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        if (whichParam == 0)
        {
            // barycentric coordinate
            silhouetteNormalResidualsJac_u_Unsafe(_mesh, 
                _V.GetVertices(), 
                _U.GetFaceIndex(k),
                _U.GetBarycentriCoordinate(k),
                _w,
                J[0]);
        }
        else
        {
            // vertex
            int vertexIndex = (*_allNarrowBands[k])[whichParam - 1];

            silhouetteNormalResidualsJac_V1_Unsafe(_mesh,
                _V.GetVertices(),
                _U.GetFaceIndex(k),
                _U.GetBarycentriCoordinate(k),
                vertexIndex,
                _w,
                J[0]);
        }
    }

    virtual bool CanBeginIteration() const
    {
        // check that vertices of each face are in the narrow bands
        const Vector<int> & L = _U.GetFaceIndices();

        for (int i=0; i < L.size(); i++)
        {
            const int * Ti = _mesh.GetTriangle(L[i]);
            const vector<int> * narrowBand = _allNarrowBands[i];

            for (int j=0; j<3; j++)
            {
                auto l = find(narrowBand->begin(), narrowBand->end(), Ti[j]);

                if (l == narrowBand->end())
                {
                    // face vertex is not in the narrow band
                    return false;
                }
            }
        }

        return true;
    }

protected:
    const VertexNode & _V;
    const BarycentricNode & _U;
    const Matrix<double> & _SN;
    const Mesh & _mesh;
    const double _w;
    const int _narrowBand;

    vector<vector<int> *> _allNarrowBands;
};

#endif
