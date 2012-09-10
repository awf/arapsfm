#ifndef __ENERGY_H__
#define __ENERGY_H__

// EnergyFactor
class EnergyFactor
{
public:
    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) const = 0;

    virtual int GetCorrespondingParam(const int k, const int i) const = 0;
    virtual int GetNumberOfMeasurements() const = 0;

    virtual void EvaluateResidual(const int k, Vector<double> & e) const = 0;
    virtual void EvaluateJacobian(const int k, const int whichParam, Vector<double> & J) const = 0;
};

// EnergyFactor_CostFunction
class EnergyFactor_CostFunction : public NLSQ_CostFunction
{
public:
    EnergyFactor_CostFunction(const & EnergyFactor parentEnergy,
                     const vector<int> * pUsedParamTypes,
                     const vector<int> * pResidualMap = nullptr)
        : _pUsedParamTypes(pUsedParamTypes),
          _pResidualMap(pResidualMap),
          NLSQ_CostFunction(*pUsedParamTypes, 3, nullptr)
                     
    virtual ~EnergyFactor_CostFunction()
    {
        delete _pUsedParamTypes;
        if (pResidualMap != nullptr)
            delete pResidualMap;
    }

    virtual int correspondingParam(const int k, const int i) const
    {
        return _parentEnergy.GetCorrespondingParam(TranslateResidualIndex(k), i);
    }

    virtual int numMeasurements() const
    {
        if (_pResidualMap == nullptr)
            return _parentEnergy.GetNumberOfMeasurements();

        return _pResidualMap->size();
    }

    virtual void evalResidual(const int k, Vector<double> & e) const
    {
        _parentEnergy.EvaluateResidual(TranslateResidualIndex(k), e);
    }

    virtual void fillJacobian(const int whichParam, const int paramIx, const int k, Matrix<double> & Jdst, const int iteration) const
    {
        _parentEnergy.EvaluateJacobian(TranslateResidualIndex(k), whichParam, Jdst);
    }

protected:
    int TranslateResidualIndex(const int k)
    {
        if (_pResidualMap == nullptr)
            return k;

        return (*_pResidualMap)[k];
    }

    const EnergyFactor & _parentEnergy;
    const vector<int> * _pUsedParamTypes;
    const vector<int> * _pResidualMap;
};

// ARAPEnergy
class ARAPEnergy : public EnergyFactor
{
public:
    ARAPEnergy(const VertexNode & V,  const RotationNode & X, const VertexNode & V1,
               const Mesh & mesh, const double w)
        : _V(V), _X(X), _V1(V), _mesh(mesh), _w(w)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) const
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_V.GetId());
        pUsedParamTypes->push_back(_V1.GetId());
        pUsedParamTypes->push_back(_X.GetId());

        costFunctions.push_back(new EnergyFactor_CostFunction(*this, pUsedParamTypes));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        switch (i)
        {
        case 0:
            return _mesh.GetHalfEdge(k, 0) + _V.GetOffset();
        case 1:
            return _mesh.GetHalfEdge(k, 1) + _V1.GetOffset();
        case 2:
            return _mesh.GetHalfEdge(k, 0) + _X.GetOffset();
        }

        assert(false);
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _mesh.GetNumberOfHalfEdges();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        int i = _mesh.GetHalfEdge(k, 0), j = _mesh.GetHalfEdge(k, 1);
        const double & edgeWeight = _mesh.GetHalfEdgeWeight(k);

        double q[4];
        quat_Unsafe(X.GetRotation(i), q);

        /* TODO */
    }

    virtual void EvaluateJacobian(const int k, const int whichParam, Vector<double> & J) const
    {
        /* TODO */
    }

protected:
    const VertexNode & _V;
    const RotationNode & _X;
    const VertexNode & _V1;

    const Mesh & _mesh;
    const double _w;
};

// LaplacianEnergy
class LaplacianEnergy : public EnergyFactor
{
public:
    LaplacianEnergy(const VertexNode & V, const Mesh & mesh, const double w)
        : _V(V), _mesh(mesh), _w(w)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) const
    {
        // map size of "inclusive one ring" to vertex id
        map<int, vector<int> *> mapIncOneRingToVertex;

        for (int i=0; i < _mesh.GetNumberOfVertices(); i++)
        {
            Vector<int> inclusiveOneRing = _mesh.GetInclusiveOneRing();
            const int n = inclusiveOneRing.size();

            auto j = mapIncOneRingToVertex.find(n);
            if (j == mapIncOneRingToVertex.end())
            {
                vector<int> * v = new vector<int>;
                v->push_back(i);
                mapIncOneRingToVertex.insert(pair<int, vector<int> * >(n, v));
            }
            else
            {
                j->second->push_back(i);
            }
        }

        // for each size of "inclusive one ring" create a cost function
        for (auto i = mapIncOneRingToVertex.begin(); i != mapIncOneRingToVertex.end(); i++)
        {
            vector<int> * pUsedParamTypes = new vector<int>;
            for (int l = 0; l < i->first; l ++)
                pUsedParamTypes->push_back(_V.GetId());

            costFunctions.push_back(new EnergyFactor_CostFunction(*this, pUsedParamTypes, i->second));
        }

        // no clean-up required (handled by EnergyFactor_CostFunction)
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        Vector<int> inclusiveOneRing = _mesh.GetInclusiveOneRing(k);

        return inclusiveOneRing[i] + _V.GetOffset();
    }

    virtual int GetNumberOfMeasurements()
    {
        return _mesh.GetNumberOfVertices();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        /* TODO */
    }
    
    virtual void EvaluateJacobian(const int k, const int whichParam, Matrix<double> & Jdst) const
    {
        /* TODO */
    }

protected:
    const VertexNode & _V;
    const Mesh & _mesh;
    const double _w;
}

// ProjectionEnergy
class ProjectionEnergy : public EnergyFactor
{
public:
    ProjectionEnergy(const VertexNode & V, const Vector<int> & C, const Matrix<double> & P,
                     const double w)
        : _V(V), _C(C), _P(P), _w(w)
    {}

    virtual void GetCostFunctions(vector<NLSQ_CostFunction *> & costFunctions) const
    {
        vector<int> * pUsedParamTypes = new vector<int>;
        pUsedParamTypes->push_back(_V.GetId());

        costFunctions.push_back(new EnergyFactor_CostFunction(*this, pUsedParamTypes));
    }

    virtual int GetCorrespondingParam(const int k, const int i) const
    {
        return _C[k] + _V.GetOffset();
    }

    virtual int GetNumberOfMeasurements() const
    {
        return _C.size();
    }

    virtual void EvaluateResidual(const int k, Vector<double> & e) const
    {
        /* TODO */
    }

    virtual void EvaluateJacobian(const int k, const in whichParam, Vector<double> & J) const
    {
        /* TODO */
    }

protected:
    const VertexNode & _V;
    const Vector<int> & _C;
    const Matrix<double> & _P;
    const double _w;
};

#endif
