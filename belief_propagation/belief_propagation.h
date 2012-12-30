/* belief_propagation.h */
#ifndef __BELIEF_PROPAGATION_H__
#define __BELIEF_PROPAGATION_H__

// Includes
#include "Math/sparse.h"
#include "Util/pyarray_conversion.h"
#include <limits>
#include <utility>
#include <algorithm>
using namespace std;

// Node
class Node
{
public:
    Node(int globalId, const Vector<int> & states)
        : _globalId(globalId), 
          _states(states),
          _stateEnergy(states.size()),
          _parent(nullptr) 
    {
        fillVector(0., _stateEnergy);
    }

    int globalId() const { return _globalId; }
    bool isLeaf() const { return _children.size() <= 0; }
    Node * parent() const { return _parent; }

    void addChild(Node * child, const int pairwiseMatrixIndex)
    {
        child->_parent = this; 

        _children.push_back(child);
        _childPairwiseMatrixIndex.push_back(pairwiseMatrixIndex);
        _childReceived.push_back(0);
        _previousChildStateIndex.push_back(Vector<int>());
    }

    void recvFromChild(Node * child)
    {
        // get local child index (linear scan)
        unsigned int i;
        for (i=0; i < _children.size(); ++i)
            if (_children[i] == child)
                break;

        // get minimum energy to each available state
        Vector<int> previousChildStateIndex(_states.size());

        for (int j=0; j < _states.size(); ++j)
        {
            double E = std::numeric_limits<double>::max();
            int previousIndex = -1;

            for (int k=0; k < child->_states.size(); ++k)
            {
                int index = _childPairwiseMatrixIndex[i];
                const PyMultipleBlockDiagonalInterface<double> & pairwiseEnergies = *((*_pairwiseEnergies)[index]);
                double kToJ = child->_stateEnergy[k] + pairwiseEnergies(child->_states[k], _states[j]);

                if (kToJ < E)
                {
                    E = kToJ;
                    previousIndex = k;
                }
            }

            _stateEnergy[j] += E;
            previousChildStateIndex[j] = previousIndex;
        }

        // save previous vector and mark the child as received
        _previousChildStateIndex[i] = std::move(previousChildStateIndex);
        _childReceived[i] = 1;

        // check if all children are received
        for (i=0; i < _childReceived.size(); ++i)
            if (!_childReceived[i])
                return;

        // propagate to the parent
        sendToParent();
    }

    void sendToParent()
    {
        // add unary energies
        for (int i=0; i < _states.size(); ++i)
            _stateEnergy[i] += (*_unaryEnergies)(_globalId, _states[i]);

        if (_parent != nullptr)
            _parent->recvFromChild(this);
    }

    int minEnergy(double * outEnergy = nullptr) const
    {
        double minEnergy = std::numeric_limits<double>::max();
        int minIndex = -1;

        for (unsigned int i = 0; i < _stateEnergy.size(); i++)
        {
            if (_stateEnergy[i] < minEnergy)
            {
                minEnergy = _stateEnergy[i];
                minIndex = i;
            }
        }

        if (outEnergy != nullptr)
            *outEnergy = minEnergy;

        // return index NOT state
        return minIndex;
    }

    void fillChildStates(Vector<int> & nodeStates) const
    {
        // get index for the current node
        int i = nodeStates[_globalId];

        // convert index into a state
        nodeStates[_globalId] = _states[i];

        // fill indices of child nodes and propagate
        for (unsigned int j=0; j < _children.size(); ++j)
        {
            const Node * child = _children[j];
            nodeStates[child->_globalId] = _previousChildStateIndex[j][i];
            child->fillChildStates(nodeStates);
        }
    }

    static void setUnaryEnergies(const PyBlockDiagonalInterface<double> * unaryEnergies)
    {
        _unaryEnergies = unaryEnergies;
    }

    static void setPairwiseEnergies(const vector<PyMultipleBlockDiagonalInterface<double> * > * pairwiseEnergies)
    {
        _pairwiseEnergies = pairwiseEnergies;
    }

protected:
    const int _globalId;
    const Vector<int> & _states;
    Vector<double> _stateEnergy;
    Node * _parent;

    std::vector<const Node *> _children;
    std::vector<int> _childPairwiseMatrixIndex;
    std::vector<unsigned char> _childReceived;
    std::vector<Vector<int>> _previousChildStateIndex;

    static const PyBlockDiagonalInterface<double> * _unaryEnergies; 
    static const vector<PyMultipleBlockDiagonalInterface<double> *> * _pairwiseEnergies;
};

const PyBlockDiagonalInterface<double> * Node::_unaryEnergies = nullptr;
const vector<PyMultipleBlockDiagonalInterface<double> *> * Node::_pairwiseEnergies = nullptr;

// tree_bp
PyObject * tree_bp(PyObject * py_unaryBlockList,
                   PyObject * py_listOfListOfPairwiseEnergyBlockLists,
                   PyObject * py_listOfColOffsets,
                   PyObject * py_listOfStates,
                   PyArrayObject * npy_nodeStateIndices,
                   PyArrayObject * npy_graphIndPtr,
                   PyArrayObject * npy_graphIndices,
                   PyArrayObject * npy_pairwiseMatrixIndices,
                   int verbose)
{
    // py_unaryBlockList -> unaryEnergies
    PyBlockDiagonalInterface<double> unaryEnergies(
        py_unaryBlockList, 
        0, 
        0, 
        numeric_limits<double>::max(),
        false);

    // py_listOfPairwiseEnergyBlockLists -> pairwiseEnergies
    Py_ssize_t l = PyList_GET_SIZE(py_listOfListOfPairwiseEnergyBlockLists);
    std::vector<PyMultipleBlockDiagonalInterface<double> *> pairwiseEnergies;
    pairwiseEnergies.reserve(l);

    for (Py_ssize_t i = 0; i < l; ++i) 
    {
        PyObject * py_listOfPairwiseEnergyBlockLists = PyList_GET_ITEM(
            py_listOfListOfPairwiseEnergyBlockLists, i);

        PyArrayObject * npy_colOffsets = (PyArrayObject *)PyList_GET_ITEM(
            py_listOfColOffsets, i);

        pairwiseEnergies.push_back(new PyMultipleBlockDiagonalInterface<double>(
            py_listOfPairwiseEnergyBlockLists,
            nullptr,
            npy_colOffsets,
            numeric_limits<double>::max(),
            true));
    }

    Node::setUnaryEnergies(&unaryEnergies);
    Node::setPairwiseEnergies(&pairwiseEnergies);

    // py_listOfStates -> listOfStates
    std::vector<Vector<int>> listOfStates = make_vectorOfVector<int>(py_listOfStates);

    // npy_nodeStateIndices -> nodeStateIndices
    PYARRAY_AS_VECTOR(int, npy_nodeStateIndices, nodeStateIndices);

    // create nodes
    unsigned int N = unaryEnergies.num_rows();

    std::vector<Node *> nodes;
    nodes.reserve(N);
    for (int i = 0; i < N; ++i)
        nodes.push_back(
            new Node(i, listOfStates[nodeStateIndices[i]]));

    // setup tree structure (Compressed Sparse Row format)
    PYARRAY_AS_VECTOR(int, npy_graphIndPtr, graphIndPtr);
    PYARRAY_AS_VECTOR(int, npy_graphIndices, graphIndices);
    PYARRAY_AS_VECTOR(int, npy_pairwiseMatrixIndices, pairwiseMatrixIndices);

    for (unsigned int m = 0; m < N; ++m)
        for (int k = graphIndPtr[m]; k < graphIndPtr[m+1]; ++k)
        {
            // edge is from m to n
            int n = graphIndices[k];
            nodes[m]->addChild(nodes[n], pairwiseMatrixIndices[k] - 1);
        }

    // perform message passing
    for (unsigned int m = 0; m < N; m++)
    {
        Node * node = nodes[m];
        if (node->isLeaf())
        {
            if (verbose >= 1)
                std::cout << "solve_tree_bp::leaf: " << node->globalId() << std::endl;

            node->sendToParent();
        }
    }

    // find the root node
    Node * root = nullptr;
    for (unsigned int m = 0; m < N; m++)
        if (nodes[m]->parent() == nullptr)
        {
            root = nodes[m];
            break;
        }

    // get minimum energy and minimum state index for the root node
    double minEnergy;
    int rootNodeStateIndex = root->minEnergy(&minEnergy);

    if (verbose >= 1)
    {
        std::cout << "solve_tree_bp::minEnergy: " << minEnergy << std::endl;
    }

    // allocate return state vector
    npy_intp dims [] = {(npy_intp)N};
    PyArrayObject * npy_nodeStates = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
    Vector<int> nodeStates(N, (int *)PyArray_DATA(npy_nodeStates));

    // fill the node states by initialise the root to its minimum index
    nodeStates[root->globalId()] = rootNodeStateIndex;
    root->fillChildStates(nodeStates);

    // free nodes
    for (auto i = nodes.begin(); i != nodes.end(); ++i)
        delete *i;

    // clean-up pairwise
    for (auto i = pairwiseEnergies.begin(); i != pairwiseEnergies.end(); ++i)
        delete *i;

    // build return
    PyObject * ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, PyFloat_FromDouble(minEnergy));
    PyTuple_SET_ITEM(ret, 1, (PyObject *)npy_nodeStates);

    return ret;
}

#endif
