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
        const int id = node->Id();

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

            _usedParameters.push_back(node->Id());

            l++;
        }

        _NLSQ_paramDesc.nParamTypes = l;
    }

    virtual ~Problem()
    {
        for (auto i = _allNodes.begin(); i != _allNodes.end(); i++)
        {
            const int n = i->second.size();

            for (int j = 0; j < n; j++)
            {
                delete i->second[j];
            }
        }
    }

protected:
    vector<int> _usedParameters;
    map<int, vector<Node *>> _allNodes;

    // NLSQ
    NLSQ_ParamDesc _NLSQ_paramDesc;
};

#endif
