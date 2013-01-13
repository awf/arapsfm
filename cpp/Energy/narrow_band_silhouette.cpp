// narrow_band_silhouette.cpp

// Includes
#include "Energy/narrow_band_silhouette.h"
#include "Math/static_linear.h"
#include <map>
#include <algorithm>
#include <set>
#include <iterator>
using namespace std;


// SilhouetteBaseEnergy
SilhouetteBaseEnergy::SilhouetteBaseEnergy(const VertexNode & V, const BarycentricNode & U,    
                                           const Mesh & mesh, const double w, const int narrowBand,
                                           const int measurementDim,
                                           const ResidualTransform * pResidualTransform)
    
        : Energy(w), _V(V), _U(U), _mesh(mesh), _narrowBand(narrowBand), _measurementDim(measurementDim), _pResidualTransform(pResidualTransform)
{}

SilhouetteBaseEnergy::~SilhouetteBaseEnergy()
{
    for (int i = 0; i < _allNarrowBands.size(); i++)
        if (_allNarrowBands[i] != nullptr) delete _allNarrowBands[i];
}

int SilhouetteBaseEnergy::GetMeasurementDim() const
{
    return _measurementDim;
}

void SilhouetteBaseEnergy::GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) 
{
    const Vector<int> & L = _U.GetFaceIndices();

    // mapping from n -> [k0, k1, ...]
    map<int, vector<int> *> narrowBandSizeToResidual;
    
    for (int i=0; i < L.size(); i++)
    {
        vector<int> * narrowBand = new vector<int>();

        for (int j=0; j<3; j++)
        {
            // source for the narrow band is taken at each vertex in the face
            int sourceVertex = _mesh.GetTriangle(L[i])[j];
            vector<int> nring = _mesh.GetNRing(sourceVertex, _narrowBand, true);

            // copy into the narrowband vector
            copy(nring.begin(), nring.end(), back_inserter(*narrowBand));
        }

        // sort inplace, remove duplicates, and save
        sort(narrowBand->begin(), narrowBand->end());
        auto it = unique(narrowBand->begin(), narrowBand->end());
        narrowBand->resize(it - narrowBand->begin());

        // save the narrow band
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

        costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, _measurementDim, i->second, _pResidualTransform));
    }
}

int SilhouetteBaseEnergy::GetCorrespondingParam(const int k, const int i) const
{
    if (i == 0)
        return k + _U.GetOffset();

    return (*_allNarrowBands[k])[i-1] + _V.GetOffset();
}

int SilhouetteBaseEnergy::GetNumberOfMeasurements() const
{
    assert(false);
    return -1;
}

bool SilhouetteBaseEnergy::CanBeginIteration() const
{
    // check that vertices of each face and one rings are in the narrow bands
    const Vector<int> & L = _U.GetFaceIndices();

    for (int i=0; i < L.size(); i++)
    {
        const int * Ti = _mesh.GetTriangle(L[i]);

        // get sorted unique vector of the adjacenct vertices
        vector<int> adjVertices;
        for (int j=0; j<3; j++)
        {
            vector<int> oneRing = _mesh.GetNRing(Ti[j], 1, true);
            copy(oneRing.begin(), oneRing.end(), back_inserter(adjVertices));
        }

        sort(adjVertices.begin(), adjVertices.end());
        auto it = unique(adjVertices.begin(), adjVertices.end());
        adjVertices.resize(it - adjVertices.begin());

        // test if the intersection between the two sorted ranges is correct
        const vector<int> * narrowBand = _allNarrowBands[i];
        vector<int> inNarrowBand;

        set_intersection(adjVertices.begin(), adjVertices.end(),
                         narrowBand->begin(), narrowBand->end(),
                         back_inserter(inNarrowBand));

        if (inNarrowBand.size() < adjVertices.size())
            return false;
    }

    return true;
}

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
SilhouetteProjectionEnergy::SilhouetteProjectionEnergy(const VertexNode & V, const BarycentricNode & U, 
                           const Matrix<double> & S, const Mesh & mesh, 
                           const double w, const int narrowBand,
                           const ResidualTransform * pResidualTransform)
    : SilhouetteBaseEnergy(V, U, mesh, w, narrowBand, 2, pResidualTransform), _S(S)
{}

void SilhouetteProjectionEnergy::EvaluateResidual(const int k, Vector<double> & e) const
{
    int faceIndex = _U.GetFaceIndex(k);
    const int * Ti = _mesh.GetTriangle(faceIndex);

    silhouetteProjResiduals_Unsafe(_V.GetVertex(Ti[0]), 
                                   _V.GetVertex(Ti[1]),
                                   _V.GetVertex(Ti[2]),
                                   _U.GetBarycentricCoordinate(k),
                                   _S[k],
                                   _w, 
                                   &e[0]);
}

void SilhouetteProjectionEnergy::EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
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
            double u = _U.GetBarycentricCoordinate(k)[0];
            silhouetteProjJac_V1l_Unsafe(u, _w, J[0]);
        }
        else if (j == Ti[1])
        {
            double u = _U.GetBarycentricCoordinate(k)[1];
            silhouetteProjJac_V1l_Unsafe(u, _w, J[0]);
        }
        else if (j == Ti[2])
        {
            double u = 1.0 - _U.GetBarycentricCoordinate(k)[0] - _U.GetBarycentricCoordinate(k)[1];
            silhouetteProjJac_V1l_Unsafe(u, _w, J[0]);
        }
        else
        {
            // other vertex in the narrow band
            fillMatrix(J, 0);
        }
    }
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

// lengthAdjustedSilhouetteProjResiduals_Unsafe
inline void lengthAdjustedSilhouetteProjResiduals_Unsafe(const double * V1i, const double * V1j, const double * V1k, const double * q, const double * S, const double & w, double * e)
{
    double V1ik[3], V1jk[3];
    subtractVectors_Static<double, 3>(V1i, V1k, V1ik);
    subtractVectors_Static<double, 3>(V1j, V1k, V1jk);

    double u[2] = {q[0] / norm_L2_Static<double, 3>(V1ik),
                   q[1] / norm_L2_Static<double, 3>(V1jk)};

    e[0] = w*(S[0] - (u[0]*(V1i[0] - V1k[0]) + u[1]*(V1j[0] - V1k[0]) + V1k[0]));
    e[1] = w*(S[1] - (u[0]*(V1i[1] - V1k[1]) + u[1]*(V1j[1] - V1k[1]) + V1k[1]));
}

// lengthAdjustedSilhouetteProjJac_V1i_Unsafe
inline void lengthAdjustedSilhouetteProjJac_V1i_Unsafe(const double * V1i, const double * V1k, const double * q, const double & w, double * J)
{
    double V1ik[3];
    subtractVectors_Static<double, 3>(V1i, V1k, V1ik);

    double nJ[9];
    normalizationJac_Unsafe(V1ik, nJ);

    scaleVectorIP_Static<double, 9>(-w * q[0], nJ);

    for (int i=0; i<6; i++)
        J[i] = nJ[i];
}

// lengthAdjustedSilhouetteProjJac_V1j_Unsafe
inline void lengthAdjustedSilhouetteProjJac_V1j_Unsafe(const double * V1j, const double * V1k, const double * q, const double & w, double * J)
{
    double V1jk[3];
    subtractVectors_Static<double, 3>(V1j, V1k, V1jk);

    double nJ[9];
    normalizationJac_Unsafe(V1jk, nJ);

    scaleVectorIP_Static<double, 9>(-w * q[1], nJ);

    for (int i=0; i<6; i++)
        J[i] = nJ[i];
}

// lengthAdjustedSilhouetteProjJac_V1k_Unsafe
inline void lengthAdjustedSilhouetteProjJac_V1k_Unsafe(const double * V1i, const double * V1j, const double * V1k, 
                                                       const double * q, const double & w, double * J)
{
    double V1ik[3], V1jk[3];
    subtractVectors_Static<double, 3>(V1i, V1k, V1ik);
    subtractVectors_Static<double, 3>(V1j, V1k, V1jk);

    double nJik[9], nJjk[9];
    normalizationJac_Unsafe(V1ik, nJik);
    normalizationJac_Unsafe(V1jk, nJjk);

    scaleVectorIP_Static<double, 9>(-w * q[0], nJik);
    scaleVectorIP_Static<double, 9>(-w * q[1], nJjk);

    for (int i=0; i<6; i++)
        J[i] = - nJik[i] - nJjk[i];

    J[0] -= 1.0;
    J[4] -= 1.0;
}


// lengthAdjustedSilhouetteProjJac_q_Unsafe
inline void lengthAdjustedSilhouetteProjJac_q_Unsafe(const double * V1i, const double * V1j, const double * V1k, 
                                                const double & w, double * J)
{
    double V1ik[3], V1jk[3], lenV1ik, lenV1jk;

    subtractVectors_Static<double, 3>(V1i, V1k, V1ik);
    lenV1ik = norm_L2_Static<double, 3>(V1ik);

    subtractVectors_Static<double, 3>(V1j, V1k, V1jk);
    lenV1jk = norm_L2_Static<double, 3>(V1jk);

    J[0] = V1ik[0] / lenV1ik;
    J[1] = V1jk[0] / lenV1jk;
    J[2] = V1ik[1] / lenV1ik;
    J[3] = V1jk[1] / lenV1jk;

    scaleVectorIP_Static<double, 4>(-w, J);
}

// LengthAdjustedSilhouetteProjectionEnergy
class LengthAdjustedSilhouetteProjectionEnergy : public SilhouetteBaseEnergy
{
public:
    LengthAdjustedSilhouetteProjectionEnergy(const VertexNode & V, const LengthAdjustedBarycentricNode & U, 
                                             const Matrix<double> & S, const Mesh & mesh, 
                                             const double w, const int narrowBand)
        : SilhouetteBaseEnergy(V, U, mesh, w, narrowBand, 2), _S(S), _Q(U)
    {}

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        int faceIndex = _U.GetFaceIndex(k);
        const int * Ti = _mesh.GetTriangle(faceIndex);

        /*
        lengthAdjustedSilhouetteProjResiduals_Unsafe(_V.GetVertex(Ti[0]), 
                                                     _V.GetVertex(Ti[1]),
                                                     _V.GetVertex(Ti[2]),
                                                     _Q.GetLengthAdjustedBarycentricCoordinate(k),
                                                     _S[k],
                                                     _w, 
                                                     &e[0]);
        */
        silhouetteProjResiduals_Unsafe(_V.GetVertex(Ti[0]), 
                                       _V.GetVertex(Ti[1]),
                                       _V.GetVertex(Ti[2]),
                                       _Q.GetBarycentricCoordinate(k),
                                       _S[k],
                                       _w, 
                                       &e[0]);

    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
    {
        const int * Ti = _mesh.GetTriangle(_Q.GetFaceIndex(k));

        if (whichParam == 0)
        {
            // barycentric coordinate
            lengthAdjustedSilhouetteProjJac_q_Unsafe(_V.GetVertex(Ti[0]), 
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
                lengthAdjustedSilhouetteProjJac_V1i_Unsafe(_V.GetVertex(Ti[0]),
                                                           _V.GetVertex(Ti[2]), 
                                                           _Q.GetLengthAdjustedBarycentricCoordinate(k),
                                                           _w, J[0]);
            }
            else if (j == Ti[1])
            {
                lengthAdjustedSilhouetteProjJac_V1j_Unsafe(_V.GetVertex(Ti[1]),
                                                           _V.GetVertex(Ti[2]), 
                                                           _Q.GetLengthAdjustedBarycentricCoordinate(k),
                                                           _w, J[0]);
            }
            else if (j == Ti[2])
            {
                lengthAdjustedSilhouetteProjJac_V1k_Unsafe(_V.GetVertex(Ti[0]),
                                                           _V.GetVertex(Ti[1]),
                                                           _V.GetVertex(Ti[2]), 
                                                           _Q.GetLengthAdjustedBarycentricCoordinate(k),
                                                           _w, J[0]);
            }
            else
            {
                // other vertex in the narrow band
                fillMatrix(J, 0);
            }
        }
    }

protected:
    const LengthAdjustedBarycentricNode & _Q;
    const Matrix<double> & _S;
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
    auto p = indexAllVertices.find(vertexIndex);
    assert(p != indexAllVertices.end());
    int desiredIndex = p->second;

    Matrix<double> outJ(3, 3, J);
    copyMatrixSlice(finalJ, 0, 3*desiredIndex, 3, 3, outJ, 0, 0);
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
SilhouetteNormalEnergy::SilhouetteNormalEnergy(const VertexNode & V, const BarycentricNode & U, 
                                               const Matrix<double> & SN, const Mesh & mesh, 
                                               const double w, const int narrowBand,
                                               const ResidualTransform * pResidualTransform)
    : SilhouetteBaseEnergy(V, U, mesh, w, narrowBand, 3, pResidualTransform), _SN(SN)
{}

void SilhouetteNormalEnergy::EvaluateResidual(const int k, Vector<double> & e) const
{
    silhouetteNormalResiduals_Unsafe(_mesh, 
                                     _V.GetVertices(),
                                     _U.GetFaceIndex(k),
                                     _U.GetBarycentricCoordinate(k),
                                     _SN[k],
                                     _w,
                                     &e[0]);
}

void SilhouetteNormalEnergy::EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
{
    if (whichParam == 0)
    {
        // barycentric coordinate
        silhouetteNormalResidualsJac_u_Unsafe(_mesh, 
            _V.GetVertices(), 
            _U.GetFaceIndex(k),
            _U.GetBarycentricCoordinate(k),
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
            _U.GetBarycentricCoordinate(k),
            vertexIndex,
            _w,
            J[0]);
    }
}

inline void lengthJac_Unsafe(const double * v, double * J)
{
    double x[5];
    x[0] = v[0]*v[0];
    x[1] = v[1]*v[1];
    x[2] = v[2]*v[2];
    x[3] = x[0] + x[1] + x[2];
    x[4] = 1.0 / sqrt(x[3]);

    J[0] = v[0] * x[4];
    J[1] = v[1] * x[4];
    J[2] = v[2] * x[4];
}

void silhouetteNormalResiduals2_Unsafe(const Mesh & mesh, const Matrix<double> & V1, 
                                       int faceIndex, const double * u_, const double * SN, 
                                       const double & w, double * e)
{
    const int * Ti = mesh.GetTriangle(faceIndex);

    double vertexNormals[3][3];
    double vertexNormalLengths[3];

    for (int i=0; i < 3; i++)
    {
        vertexNormal_Unsafe(mesh, V1, Ti[i], vertexNormals[i]);
        vertexNormalLengths[i] = norm_L2_Static<double, 3>(vertexNormals[i]);
        scaleVectorIP_Static<double, 3>(1.0 / vertexNormalLengths[i], vertexNormals[i]);
    }

    double u[3] = { u_[0], u_[1], 1.0 - u_[0] - u_[1] };

    double normal[3];
    makeTriInterpolatedVector_Static<double, 3>(u, vertexNormals[0], vertexNormals[1], vertexNormals[2], normal);
    normalizeVector_Static<double, 3>(normal);

    double weightedLengths = (u[0] * vertexNormalLengths[0] +
                              u[1] * vertexNormalLengths[1] +
                              u[2] * vertexNormalLengths[2]);

    e[0] = w * weightedLengths * (SN[0] - normal[0]);
    e[1] = w * weightedLengths * (SN[1] - normal[1]);
    e[2] = w * weightedLengths * (- normal[2]);
}

void silhouetteNormalResiduals2Jac_V1_Unsafe(const Mesh & mesh, const Matrix<double> & V1, 
                                             int faceIndex, const double * u_, 
                                             const double * SN,
                                             int vertexIndex, const double & w, double * J)
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

    // build an index mapping for `allVertices` (which is ordered)
    std::map<int, int> indexAllVertices;
    auto it = allVertices.begin();
    for (int l=0; it != allVertices.end(); l++, it++)
        indexAllVertices.insert(std::pair<int, int>(*it, l));

    // construct the un-normalised normal estimate jacobian for **all** vertices in the 
    // extended one-ring
    double normalisedVertexNormals[3][3];
    Matrix<double> unJ(3, 3*allVertices.size(), 0.);

    Matrix<double> wJ(1, 3*allVertices.size(), 0.);
    double weightedLengths = 0.;

    for (int i = 0; i < 3; i++)
    {
        // get the (un-normalised) vertex normal
        double * vertexNormal = normalisedVertexNormals[i];
        vertexNormal_Unsafe(mesh, V1, Ti[i], vertexNormal);

        // get the normalisation jacobian
        Matrix<double> normalizationJac(3,3);
        normalizationJac_Unsafe(vertexNormal, normalizationJac[0]);

        // get the length jacobian
        Matrix<double> lengthJac(1,3);
        lengthJac_Unsafe(vertexNormal, lengthJac[0]);

        // get the Jacobian for the given vertex (column ordering same as `oneRings[i]`)
        Matrix<double> unVertexNormalJ = vertexNormalJac(mesh, V1, Ti[i]);

        // apply the normalisation Jacobian (chain rule)
        Matrix<double> vertexJ(3, unVertexNormalJ.num_cols());
        multiply_A_B(normalizationJac, unVertexNormalJ, vertexJ);

        // apply the length Jacobian (chain rule)
        Matrix<double> lengthJ(1, unVertexNormalJ.num_cols());
        multiply_A_B(lengthJac, unVertexNormalJ, lengthJ);

        // add and scale the columns into the main Jacobian matrices
        auto it = oneRings[i].begin();
        for (int l=0; it != oneRings[i].end(); it++, l++)
        {
            auto p = indexAllVertices.find(*it);
            assert(p != indexAllVertices.end());
            int m = p->second;

            for (int k=0; k<3; k++)
            {
                for (int r=0; r<3; r++)
                    unJ[r][3*m + k] += u[i] * vertexJ[r][3*l + k];

                wJ[0][3*m + k] += u[i] * lengthJ[0][3*l + k];
            }

        }

        // get the vertex normal length
        double vertexNormalLength = norm_L2_Static<double, 3>(vertexNormal);
        weightedLengths += u[i] * vertexNormalLength;

        // normalise the vertex normal
        scaleVectorIP_Static<double, 3>(1.0 / vertexNormalLength, vertexNormal);
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

    // apply to the Jacobian for p2 (chain rule)
    Matrix<double> Jp2(3, unJ.num_cols());
    multiply_A_B(normalizationJac, unJ, Jp2);
    scaleMatrixIP(-1.0, Jp2);

    // construct the normal vector
    normalizeVector_Static<double, 3>(normal);

    // construct the residual
    double residual[3];
    residual[0] = SN[0] - normal[0];
    residual[1] = SN[1] - normal[1];
    residual[2] =  - normal[2];

    // construct the final Jacobian
    Matrix<double> finalJ(3, unJ.num_cols());
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < unJ.num_cols(); c++)
            finalJ[r][c] = w * (wJ[0][c] * residual[r] + Jp2[r][c] * weightedLengths);

    // return slice for the vertex of interest
    auto p = indexAllVertices.find(vertexIndex);
    assert(p != indexAllVertices.end());
    int desiredIndex = p->second;

    Matrix<double> outJ(3, 3, J);
    copyMatrixSlice(finalJ, 0, 3*desiredIndex, 3, 3, outJ, 0, 0);
}

void silhouetteNormalResiduals2Jac_u_Unsafe(const Mesh & mesh, const Matrix<double> & V1, 
                                            int faceIndex, const double * u_, const double * SN, 
                                            const double & w, double * J_)
{
    const int * Ti = mesh.GetTriangle(faceIndex);

    // get the normalised vertex normals and the vertex normal lengths
    double vertexNormals[3][3];
    double vertexNormalLengths[3];

    for (int i=0; i < 3; i++)
    {
        double * vertexNormal = vertexNormals[i];
        vertexNormal_Unsafe(mesh, V1, Ti[i], vertexNormal);

        vertexNormalLengths[i] = norm_L2_Static<double, 3>(vertexNormal);
        scaleVectorIP_Static<double, 3>(1.0 / vertexNormalLengths[i], vertexNormal);
    }

    // get the weighted lengths
    double u[3] = { u_[0], u_[1], 1.0 - u_[0] - u_[1] };
    double weightedLengths = (u[0] * vertexNormalLengths[0] +
                              u[1] * vertexNormalLengths[1] +
                              u[2] * vertexNormalLengths[2]);

    double unJ[3][2];
    for (int i=0; i < 3; i++)
        for (int j=0; j < 2; j++)
            unJ[i][j] = vertexNormals[j][i] - vertexNormals[2][i];

    double unJ1[2];
    unJ1[0] = vertexNormalLengths[0] - vertexNormalLengths[2];
    unJ1[1] = vertexNormalLengths[1] - vertexNormalLengths[2];

    // get the blended normal
    double normal[3];

    makeTriInterpolatedVector_Static<double, 3>(u, 
            vertexNormals[0],  
            vertexNormals[1], 
            vertexNormals[2], normal);

    // get the normalisation Jacobian
    double normalizationJac[9];
    normalizationJac_Unsafe(normal, normalizationJac);

    // get the p2 Jacobian
    Matrix<double> Jp2(3, 2);
    multiply_A_B_Static<double, 3, 3, 2>(normalizationJac, unJ[0], Jp2[0]);
    scaleVectorIP_Static<double, 6>(-1.0 , Jp2[0]);

    // construct the normal vector
    normalizeVector_Static<double, 3>(normal);

    // construct the residual
    double residual[3];
    residual[0] = SN[0] - normal[0];
    residual[1] = SN[1] - normal[1];
    residual[2] =  - normal[2];

    // construct the final Jacobian
    Matrix<double> J(3, 2, J_);

    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 2; c++)
            J[r][c] = w * (unJ1[c] * residual[r] + Jp2[r][c] * weightedLengths);
}

// SilhouetteNormalEnergy2
SilhouetteNormalEnergy2::SilhouetteNormalEnergy2(const VertexNode & V, const BarycentricNode & U, 
                                                 const Matrix<double> & SN, const Mesh & mesh, 
                                                 const double w, const int narrowBand,
                                                 const ResidualTransform * pResidualTransform)
    : SilhouetteBaseEnergy(V, U, mesh, w, narrowBand, 3, pResidualTransform), _SN(SN)
{}

// SilhouetteNormalEnergy2
void SilhouetteNormalEnergy2::EvaluateResidual(const int k, Vector<double> & e) const
{
    silhouetteNormalResiduals2_Unsafe(_mesh, 
                                      _V.GetVertices(),
                                      _U.GetFaceIndex(k),
                                      _U.GetBarycentricCoordinate(k),
                                      _SN[k],
                                      _w,
                                      &e[0]);
}

void SilhouetteNormalEnergy2::EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const
{
    if (whichParam == 0)
    {
        // barycentric coordinate
        silhouetteNormalResiduals2Jac_u_Unsafe(_mesh, 
            _V.GetVertices(), 
            _U.GetFaceIndex(k),
            _U.GetBarycentricCoordinate(k),
            _SN[k],
            _w,
            J[0]);
    }
    else
    {
        // vertex
        int vertexIndex = (*_allNarrowBands[k])[whichParam - 1];

        silhouetteNormalResiduals2Jac_V1_Unsafe(_mesh,
            _V.GetVertices(),
            _U.GetFaceIndex(k),
            _U.GetBarycentricCoordinate(k),
            _SN[k],
            vertexIndex,
            _w,
            J[0]);
    }
}

