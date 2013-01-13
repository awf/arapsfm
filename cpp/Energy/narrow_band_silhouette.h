#ifndef __NARROW_BAND_SILHOUETTE_H__
#define __NARROW_BAND_SILHOUETTE_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Solve/node.h"
#include "Energy/energy.h"
#include "Geometry/mesh.h"
using namespace std;

// SilhouetteBaseEnergy
class SilhouetteBaseEnergy : public Energy
{
public:
    SilhouetteBaseEnergy(const VertexNode & V, const BarycentricNode & U,    
                         const Mesh & mesh, const double w, const int narrowBand,
                         const int measurementDim,
                         const ResidualTransform * pResidualTransform = nullptr);

    virtual ~SilhouetteBaseEnergy();
    virtual int GetMeasurementDim() const;
    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions);
    virtual int GetCorrespondingParam(const int k, const int i) const;
    virtual int GetNumberOfMeasurements() const;
    virtual bool CanBeginIteration() const;

protected:
    const VertexNode & _V;
    const BarycentricNode & _U;
    
    const int _narrowBand;

    const Mesh & _mesh;

    vector<vector<int> *> _allNarrowBands;

    const int _measurementDim;

    const ResidualTransform * _pResidualTransform;
};

// SilhouetteProjectionEnergy
class SilhouetteProjectionEnergy : public SilhouetteBaseEnergy
{
public:
    SilhouetteProjectionEnergy(const VertexNode & V, const BarycentricNode & U, 
                               const Matrix<double> & S, const Mesh & mesh, 
                               const double w, const int narrowBand,
                               const ResidualTransform * pResidualTransform = nullptr);

    virtual void EvaluateResidual(const int k, Vector<double> & e) const;
    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const;

protected:
    const Matrix<double> & _S;
};


// SilhouetteNormalEnergy
class SilhouetteNormalEnergy : public SilhouetteBaseEnergy
{
public:
    SilhouetteNormalEnergy(const VertexNode & V, const BarycentricNode & U, 
                           const Matrix<double> & SN, const Mesh & mesh, 
                           const double w, const int narrowBand,
                           const ResidualTransform * pResidualTransform = nullptr);

    virtual void EvaluateResidual(const int k, Vector<double> & e) const;
    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const;

protected:
    const Matrix<double> & _SN;
};

// SilhouetteNormalEnergy2
class SilhouetteNormalEnergy2 : public SilhouetteBaseEnergy
{
public:
    SilhouetteNormalEnergy2(const VertexNode & V, const BarycentricNode & U, 
                            const Matrix<double> & SN, const Mesh & mesh, 
                            const double w, const int narrowBand,
                            const ResidualTransform * pResidualTransform = nullptr);

    virtual void EvaluateResidual(const int k, Vector<double> & e) const;
    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & J) const;

protected:
    const Matrix<double> & _SN;
};

#endif
