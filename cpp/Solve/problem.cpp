#include "problem.h"
#include "optimiser.h"
#include <cmath>

#include <iostream>
using namespace std;

Problem::Problem()
    : _maxJteStore(0)
{}

void Problem::AddNode(Node * node, bool isFixed)
{
    if (isFixed || node->GetCount() == 0)
        return AddFixedNode(node);

    const int id = node->TypeId();

    auto i = _allNodes.find(id);
    if (i == _allNodes.end())
    {
        vector<Node *> nodePointers;
        nodePointers.push_back(node);
        _allNodes.insert(pair<int, vector<Node *>>(id, nodePointers));
    }
    else
    {
        const Node * lastNode = i->second.back();
        node->SetOffset(lastNode->GetOffset() + lastNode->GetCount());
        i->second.push_back(node);
    }
}

void Problem::AddFixedNode(Node * node)
{
    _fixedNodes.push_back(node);
}

void Problem::AddEnergy(Energy * energy)
{
    _allEnergies.push_back(energy);
}

void Problem::InitialiseParamDesc()
{
    _usedParameters.clear();

    int l = 0;
    for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
    {
        const Node * node = i->second.back();

        _NLSQ_paramDesc.dimension[l] = node->Dimension();
        _NLSQ_paramDesc.count[l] = node->GetOffset() + node->GetCount();
        _NLSQ_paramDesc.preconditioner[l] = node->GetPreconditioner();

        _usedParameters.push_back(node->TypeId());

        for (int j = 0; j < i->second.size(); j++)
            i->second[j]->SetParamId(l);

        l++;
    }

    _NLSQ_paramDesc.nParamTypes = l;
}

void Problem::InitialiseCostFunctions()
{
    _costFunctions.clear(); 
    for (auto i = _allEnergies.begin(); i != _allEnergies.end(); i++)
        (*i)->GetCostFunctions(_costFunctions);
}

bool Problem::BeginIteration(const int currentIteration)
{
    for (auto energy = _allEnergies.begin(); energy != _allEnergies.end(); energy++)
        if (!(*energy)->CanBeginIteration()) 
            return false;

    return true;
}

int Problem::Minimise(const OptimiserOptions & options)
{
    InitialiseParamDesc();
    InitialiseCostFunctions();

    Optimiser opt = Optimiser(_NLSQ_paramDesc, _costFunctions, *this);

    opt.maxIterations = options.maxIterations;
    opt.minIterations = options.minIterations;
    opt.tau = options.tau;
    opt.lambda = options.lambda;
    opt.gradientThreshold = options.gradientThreshold;
    opt.updateThreshold = options.updateThreshold;
    opt.improvementThreshold = options.improvementThreshold;
    opt.useAsymmetricLambda = options.useAsymmetricLambda;

    optimizerVerbosenessLevel = options.verbosenessLevel;

    if (optimizerVerbosenessLevel > 0)
    {
        cout << "opt.maxIterations: " << opt.maxIterations << endl;
        cout << "opt.minIterations: " << opt.minIterations << endl;
        cout << "opt.tau: " << opt.tau << endl;
        cout << "opt.lambda: " << opt.lambda << endl;
        cout << "opt.gradientThreshold: " << opt.gradientThreshold << endl;
        cout << "opt.updateThreshold: " << opt.updateThreshold << endl;
        cout << "opt.improvementThreshold: " << opt.improvementThreshold << endl;
        cout << "opt.useAsymmetricLambda: " << opt.useAsymmetricLambda << endl;
    }

    opt.minimize();

    return opt.status;
}

void Problem::UpdateParameter(const int paramType, const VectorArrayAdapter<double> & delta)
{
    int typeId = _usedParameters[paramType];
    vector<Node *> & nodes = _allNodes[typeId];

    for (int i=0; i < nodes.size(); i++)
    {
        Node * node = nodes[i];
        const Vector<double> & deltaStart = delta[node->GetOffset()];

        VectorArrayAdapter<double> deltaSlice(
            node->GetCount(), 
            node->Dimension(), 
            const_cast<double *>(deltaStart.begin()));

        node->Update(deltaSlice);
    }
}

void Problem::Save()
{
    for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
        for (auto j = i->second.begin(); j != i->second.end(); j++)
            (*j)->Save();
}

void Problem::Restore()
{
    for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
        for (auto j = i->second.begin(); j != i->second.end(); j++)
            (*j)->Restore();
}

double Problem::GetParameterLength() const
{
    double squareLength = 0.;
    for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
        for (auto j = i->second.begin(); j != i->second.end(); j++)
            squareLength += (*j)->SquareLength();

    return sqrt(squareLength);
}

void Problem::EvaluateJteCallback(const Vector<double> & Jt_e)
{
    if (_maxJteStore > 0)
    {
        while (_storedJte.size() > _maxJteStore)
        {
            Vector<double> * oldJt_e = _storedJte.front();
            delete oldJt_e;

            _storedJte.pop_front();
        }

        Vector<double> * copyJt_e = new Vector<double>(Jt_e.size());
        copyVector(Jt_e, *copyJt_e);

        _storedJte.push_back(copyJt_e);
    }
}

Problem::~Problem()
{
    for (int i = 0; i < _allEnergies.size(); i++)
        delete _allEnergies[i];

    for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
        for (int j = 0; j < i->second.size(); j++)
            delete i->second[j];

    for (int i = 0; i < _fixedNodes.size(); i++)
        delete _fixedNodes[i];

    for (int i = 0; i < _storedJte.size(); i++)
        delete _storedJte[i];
}

