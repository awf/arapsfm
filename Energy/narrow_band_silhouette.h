#ifndef __NARROW_BAND_SILHOUETTE_H__
#define __NARROW_BAND_SILHOUETTE_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/mesh.h"

#include <map>
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
        const Matrix<double> & U = _U.GetBarycentricCoordinates();
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

protected:
    const VertexNode & _V;
    const BarycentricNode & _U;
    const Matrix<double> & _S;
    const Mesh & _mesh;
    const double _w;
    const int _narrowBand;

    vector<vector<int> *> _allNarrowBands;
};

#endif
