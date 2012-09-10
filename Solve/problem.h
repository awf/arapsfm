#ifndef __PROBLEM_H__
#define __PROBLEM_H__

#include <vector>
#include <map>
#include <vector>
#include <utility>
using namespace std;

#include "node.h"

// Problem
class Problem
{
public:
    void AddNode(Node * node)
    {
        const int id = node->Id();

        auto i = allNodes.find(id);
        if (i == allNodes.end())
        {
            vector<Node *> nodePointers;
            nodePointers.push_back(node);
            allNodes.insert(pair<int, vector<Node *>>(id, nodePointers));
        }
        else
        {
            const Node * lastNode = i->second.back();
            node->SetOffset(lastNode->GetOffset() + lastNode->GetCount());
            i->second.push_back(node);
        }
    }

    virtual ~Problem()
    {
        for (auto i = allNodes.begin(); i != allNodes.end(); i++)
        {
            const int n = i->second.size();

            for (int j = 0; j < n; j++)
            {
                delete i->second[j];
            }
        }
    }

protected:
    map<int, vector<Node *>> allNodes;
};

#endif
