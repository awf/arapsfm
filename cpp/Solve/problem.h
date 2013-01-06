#ifndef __PROBLEM_H__
#define __PROBLEM_H__

#include <vector>
#include <map>
#include <utility>
#include <deque>
using namespace std;

#include "Solve/node.h"
#include "Solve/optimiser_options.h"
#include "Energy/energy.h"

// Problem
class Problem
{
public:
    Problem();
    virtual void AddNode(Node * node, bool isFixed=false);
    virtual void AddFixedNode(Node * node);
    virtual void AddEnergy(Energy * energy);

    virtual void InitialiseParamDesc();
    virtual void InitialiseCostFunctions();

    virtual int Minimise(const OptimiserOptions & options);

    virtual bool BeginIteration(const int currentIteration);

    virtual void UpdateParameter(const int paramType, const VectorArrayAdapter<double> & delta);

    virtual void Save();
    virtual void Restore();

    virtual double GetParameterLength() const;

    void SetMaximumJteToStore(int maxJteStore) { _maxJteStore = maxJteStore; }
    const deque<Vector<double> *> GetStoredJte() const { return _storedJte; }

    virtual void EvaluateJteCallback(const Vector<double> & Jt_e);
    virtual ~Problem();

protected:
    vector<int> _usedParameters;
    map<int, vector<Node *>> _allNodes;
    vector<Node *> _fixedNodes;
    vector<Energy *> _allEnergies;

    deque<Vector<double> *> _storedJte;
    int _maxJteStore;

    // NLSQ
    NLSQ_ParamDesc _NLSQ_paramDesc;
    vector<NLSQ_CostFunction *> _costFunctions;
};


#endif
