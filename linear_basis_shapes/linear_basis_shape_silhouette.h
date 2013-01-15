#ifndef __LINEAR_BASIS_SHAPE_SILHOUETTE_H__
#define __LINEAR_BASIS_SHAPE_SILHOUETTE_H__

#include "linear_basis_shape.h"
#include "Energy/narrow_band_silhouette.h"
#include <utility>
using namespace std;

// TODO: Clean-up ...

// LinearBasisShapeSilhouette_GetNarrowBands
void LinearBasisShapeSilhouette_GetNarrowBands(
    const Mesh & mesh,
    const Vector<int> & L,
    int _narrowBand,
    vector<vector<int> *> * allNarrowBands,
    map<int, vector<int> *> * narrowBandSizeToResidual)
{
    // mapping from n -> [k0, k1, ...]
    for (int i=0; i < L.size(); i++)
    {
        vector<int> * narrowBand = new vector<int>();

        for (int j=0; j<3; j++)
        {
            // source for the narrow band is taken at each vertex in the face
            int sourceVertex = mesh.GetTriangle(L[i])[j];
            vector<int> nring = mesh.GetNRing(sourceVertex, _narrowBand, true);

            // copy into the narrowband vector
            copy(nring.begin(), nring.end(), back_inserter(*narrowBand));
        }

        // sort inplace, remove duplicates, and save
        sort(narrowBand->begin(), narrowBand->end());
        auto it = unique(narrowBand->begin(), narrowBand->end());
        narrowBand->resize(it - narrowBand->begin());

        // save the narrow band
        allNarrowBands->push_back(narrowBand);

        // save the mapping from size to residual index
        int n = narrowBand->size();

        auto j = narrowBandSizeToResidual->find(n);
        if (j == narrowBandSizeToResidual->end())
        {
            vector<int> * residualMap = new vector<int>(1, i);
            narrowBandSizeToResidual->insert(pair<int, vector<int> *>(n, residualMap));
        }
        else
        {
            j->second->push_back(i);
        }
    }
}

// LinearBasisShapeSilhouetteProjectionEnergy
class LinearBasisShapeSilhouetteProjectionEnergy : public SilhouetteProjectionEnergy
{
public:
    LinearBasisShapeSilhouetteProjectionEnergy(const LinearBasisShapeNode & V, 
        const BarycentricNode & U, 
        const Matrix<double> & S, const Mesh & mesh, 
        const double w, 
        const int narrowBand,
        const ResidualTransform * pResidualTransform = nullptr)
        : SilhouetteProjectionEnergy(V, U, S, mesh, w, narrowBand, pResidualTransform), __V(V)
    {
        __jacobianCache.resize(_S.num_rows());
    }

    void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        // mapping from n -> [k0, k1, ...]
        map<int, vector<int> *> narrowBandSizeToResidual;
        LinearBasisShapeSilhouette_GetNarrowBands(_mesh, _U.GetFaceIndices(), _narrowBand, 
                                                  &_allNarrowBands, &narrowBandSizeToResidual);
        
        // construct cost functions
        for (auto i = narrowBandSizeToResidual.begin(); i != narrowBandSizeToResidual.end(); i++)
        {
            int n = i->first;

            vector<int> * pUsedParamTypes = new vector<int>;
            pUsedParamTypes->push_back(_U.GetParamId());

            for (int j = 0; j < n; j++)
                __V.AddVertexUsedParamTypes(pUsedParamTypes);

            __V.AddGlobalUsedParamTypes(pUsedParamTypes);

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 
                _measurementDim, i->second, _pResidualTransform));
        }
    }

    int GetCorrespondingParam(const int k, const int l) const
    {
        int whichParam = l;

        if (whichParam == 0)
            return k + _U.GetOffset();

        whichParam -= 1;
        int p;

        for (int i = 0; i < _allNarrowBands[k]->size(); i++)
        {
            p = __V.GetVertexParam(whichParam, (*_allNarrowBands[k])[i]);
            if (p >= 0) return p;
        }

        p = __V.GetScaleParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetGlobalRotationParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetDisplacmentParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetCoefficientParam(whichParam);
        if (p >= 0) return p;

        assert(false);
    }

    bool CanBeginIteration() const
    {
        if (SilhouetteProjectionEnergy::CanBeginIteration())
        {
            for (auto i = __jacobianCache.begin(); i != __jacobianCache.end(); i++)
            {
                const_cast< vector<Matrix<double>> *>(&(*i))->clear();
            }

            return true;
        }

        return false;
    }

    void EvaluateJacobian(const int k, const int l, Matrix<double> & J) const
    {
        int whichParam = l;

        if (whichParam == 0)
            return SilhouetteProjectionEnergy::EvaluateJacobian(k, 0, J);

        whichParam -= 1;

        const vector<int> & narrowBand = *_allNarrowBands[k];

        for (int i = 0; i < _allNarrowBands[k]->size(); i++)
        {
            if (__V.GetVertexParam(whichParam, narrowBand[i]) >= 0)
            {
                Matrix<double> Jr(2, 3);
                SilhouetteProjectionEnergy::EvaluateJacobian(k, i + 1, Jr);

                Matrix<double> JV(3, 3);
                __V.VertexJacobian(whichParam, narrowBand[i], JV);
                multiply_A_B(Jr, JV, J);
                return;
            }
        }

        if (__jacobianCache[k].size() == 0)
        {
            vector<Matrix<double>> * j = const_cast< vector<Matrix<double>> *>(&__jacobianCache[k]);

            for (int i = 0; i < _allNarrowBands[k]->size(); i++)
            {
                Matrix<double> Jr(2, 3);
                SilhouetteProjectionEnergy::EvaluateJacobian(k, i + 1, Jr);
                j->push_back(move(Jr));
            }
        }

        const vector<Matrix<double>> & all_Jr = __jacobianCache[k];

        if (__V.GetScaleParam(whichParam) >= 0)
        {
            fillMatrix(J, 0.);

            Matrix<double> Jt(2, 1);
            Matrix<double> Js(3, 1);

            for (int i = 0; i < all_Jr.size(); i++)
            {
                __V.ScaleJacobian(narrowBand[i], Js);
                multiply_A_B(all_Jr[i], Js, Jt);
                addMatricesIP(Jt, J);
            }
            return;
        }

        if (__V.GetGlobalRotationParam(whichParam) >= 0)
        {
            fillMatrix(J, 0.);

            Matrix<double> Jt(2, 3);
            Matrix<double> JXg(3, 3);

            for (int i = 0; i < all_Jr.size(); i++)
            {
                __V.GlobalRotationJacobian(narrowBand[i], JXg);
                multiply_A_B(all_Jr[i], JXg, Jt);
                addMatricesIP(Jt, J);
            }
            return;
        }

        if (__V.GetDisplacmentParam(whichParam) >= 0)
        {
            fillMatrix(J, 0.);

            Matrix<double> Jt(2, 3);
            Matrix<double> JVd(3, 3);

            for (int i = 0; i < all_Jr.size(); i++)
            {
                __V.DisplacementJacobian(JVd);
                multiply_A_B(all_Jr[i], JVd, Jt);
                addMatricesIP(Jt, J);
            }
            return;
        }

        if (__V.GetCoefficientParam(whichParam) >= 0)
        {
            fillMatrix(J, 0.);

            Matrix<double> Jt(2, 3);
            Matrix<double> Jy(3, 1);

            for (int i = 0; i < all_Jr.size(); i++)
            {
                __V.CoefficientJacobian(whichParam++, narrowBand[i], Jy);
                multiply_A_B(all_Jr[i], Jy, Jt);
                addMatricesIP(Jt, J);
            }
            return;
        }

        assert(false);
    }

protected:
    const LinearBasisShapeNode & __V;
    vector<vector<Matrix<double>>> __jacobianCache;
};

// LinearBasisShapeSilhouetteNormalEnergy2
class LinearBasisShapeSilhouetteNormalEnergy2 : public SilhouetteNormalEnergy2
{
public:
    LinearBasisShapeSilhouetteNormalEnergy2(const LinearBasisShapeNode & V, 
        const BarycentricNode & U, 
        const Matrix<double> & SN, const Mesh & mesh, 
        const double w, 
        const int narrowBand,
        const ResidualTransform * pResidualTransform = nullptr)
        : SilhouetteNormalEnergy2(V, U, SN, mesh, w, narrowBand, pResidualTransform), __V(V)
    {
        __jacobianCache.resize(_SN.num_rows());
    }

    bool CanBeginIteration() const
    {
        if (SilhouetteNormalEnergy2::CanBeginIteration())
        {
            for (auto i = __jacobianCache.begin(); i != __jacobianCache.end(); i++)
            {
                const_cast< vector<Matrix<double>> *>(&(*i))->clear();
            }

            return true;
        }

        return false;
    }

    void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions)
    {
        // mapping from n -> [k0, k1, ...]
        map<int, vector<int> *> narrowBandSizeToResidual;
        LinearBasisShapeSilhouette_GetNarrowBands(_mesh, _U.GetFaceIndices(), _narrowBand, 
                                                  &_allNarrowBands, &narrowBandSizeToResidual);
        
        // construct cost functions
        for (auto i = narrowBandSizeToResidual.begin(); i != narrowBandSizeToResidual.end(); i++)
        {
            int n = i->first;

            vector<int> * pUsedParamTypes = new vector<int>;
            pUsedParamTypes->push_back(_U.GetParamId());

            for (int j = 0; j < n; j++)
                __V.AddVertexUsedParamTypes(pUsedParamTypes);

            __V.AddGlobalUsedParamTypes(pUsedParamTypes);

            costFunctions.push_back(new Energy_CostFunction(*this, pUsedParamTypes, 
                _measurementDim, i->second, _pResidualTransform));
        }
    }

    int GetCorrespondingParam(const int k, const int l) const
    {
        int whichParam = l;

        if (whichParam == 0)
            return k + _U.GetOffset();

        whichParam -= 1;
        int p;

        for (int i = 0; i < _allNarrowBands[k]->size(); i++)
        {
            p = __V.GetVertexParam(whichParam, (*_allNarrowBands[k])[i]);
            if (p >= 0) return p;
        }

        p = __V.GetScaleParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetGlobalRotationParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetDisplacmentParam(whichParam);
        if (p >= 0) return p;

        p = __V.GetCoefficientParam(whichParam);
        if (p >= 0) return p;

        assert(false);
    }

    void EvaluateJacobian(const int k, const int l, Matrix<double> & J) const
    {
        int whichParam = l;

        if (whichParam == 0)
            return SilhouetteNormalEnergy2::EvaluateJacobian(k, 0, J);

        whichParam -= 1;

        const vector<int> & narrowBand = *_allNarrowBands[k];

        for (int i = 0; i < _allNarrowBands[k]->size(); i++)
        {
            if (__V.GetVertexParam(whichParam, narrowBand[i]) >= 0)
            {
                Matrix<double> Jr(3, 3);
                SilhouetteNormalEnergy2::EvaluateJacobian(k, i + 1, Jr);

                Matrix<double> JV(3, 3);
                __V.VertexJacobian(whichParam, narrowBand[i], JV);
                multiply_A_B(Jr, JV, J);
                return;
            }
        }

        if (__jacobianCache[k].size() == 0)
        {
            vector<Matrix<double>> * j = const_cast< vector<Matrix<double>> *>(&__jacobianCache[k]);

            for (int i = 0; i < _allNarrowBands[k]->size(); i++)
            {
                Matrix<double> Jr(3, 3);
                SilhouetteNormalEnergy2::EvaluateJacobian(k, i + 1, Jr);
                j->push_back(move(Jr));
            }
        }

        const vector<Matrix<double>> & all_Jr = __jacobianCache[k];


        if (__V.GetScaleParam(whichParam) >= 0)
        {
            fillMatrix(J, 0.);

            Matrix<double> Jt(3, 1);
            Matrix<double> Js(3, 1);

            for (int i = 0; i < all_Jr.size(); i++)
            {
                __V.ScaleJacobian(narrowBand[i], Js);
                multiply_A_B(all_Jr[i], Js, Jt);
                addMatricesIP(Jt, J);
            }
            return;
        }

        if (__V.GetGlobalRotationParam(whichParam) >= 0)
        {
            fillMatrix(J, 0.);

            Matrix<double> Jt(3, 3);
            Matrix<double> JXg(3, 3);

            for (int i = 0; i < all_Jr.size(); i++)
            {
                __V.GlobalRotationJacobian(narrowBand[i], JXg);
                multiply_A_B(all_Jr[i], JXg, Jt);
                addMatricesIP(Jt, J);
            }
            return;
        }

        if (__V.GetDisplacmentParam(whichParam) >= 0)
        {
            fillMatrix(J, 0.);

            Matrix<double> Jt(3, 3);
            Matrix<double> JVd(3, 3);

            for (int i = 0; i < all_Jr.size(); i++)
            {
                __V.DisplacementJacobian(JVd);
                multiply_A_B(all_Jr[i], JVd, Jt);
                addMatricesIP(Jt, J);
            }
            return;
        }

        if (__V.GetCoefficientParam(whichParam) >= 0)
        {
            fillMatrix(J, 0.);

            Matrix<double> Jt(3, 3);
            Matrix<double> Jy(3, 1);

            for (int i = 0; i < all_Jr.size(); i++)
            {
                __V.CoefficientJacobian(whichParam++, narrowBand[i], Jy);
                multiply_A_B(all_Jr[i], Jy, Jt);
                addMatricesIP(Jt, J);
            }
            return;
        }

        assert(false);
    }

protected:
    const LinearBasisShapeNode & __V;
    vector<vector<Matrix<double>>> __jacobianCache;
};

#endif
