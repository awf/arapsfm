#ifndef __PROBLEM_H__
#define __PROBLEM_H__

#include <vector>
#include <map>
#include <vector>
#include <utility>
using namespace std;

#include "node.h"
#include "Energy/energy.h"

// Problem
class Problem
{
public:
    void AddNode(Node * node)
    {
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

    void AddEnergy(Energy * energy)
    {
        _allEnergies.push_back(energy);
    }

    void InitialiseParamDesc()
    {
        _usedParameters.clear();

        int l = 0;
        for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
        {
            const Node * node = i->second.back();

            _NLSQ_paramDesc.dimension[l] = node->Dimension();
            _NLSQ_paramDesc.count[l] = node->GetOffset() + node->GetCount();

            _usedParameters.push_back(node->TypeId());

            for (int j = 0; j < i->second.size(); j++)
                i->second[j]->SetParamId(l);

            l++;
        }

        _NLSQ_paramDesc.nParamTypes = l;
    }

    void UpdateParameter(const int paramType, const VectorArrayAdapter<double> & delta)
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

    void Save()
    {
        for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
            for (auto j = i->second.begin(); j != i->second.end(); j++)
                (*j)->Save();
    }

    void Restore()
    {
        for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
            for (auto j = i->second.begin(); j != i->second.end(); j++)
                (*j)->Restore();

    }

    virtual ~Problem()
    {
        for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
        {
            const int n = i->second.size();

            for (int j = 0; j < n; j++)
                delete i->second[j];
        }

        const int n = _allEnergies.size();

        for (int i = 0; i < n; i++)
        {
            delete _allEnergies[i];
        }
    }

protected:
    vector<int> _usedParameters;
    map<int, vector<Node *>> _allNodes;
    vector<Energy *> _allEnergies;

    // NLSQ
    NLSQ_ParamDesc _NLSQ_paramDesc;
};

#endif
