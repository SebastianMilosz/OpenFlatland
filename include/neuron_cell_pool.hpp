#ifndef NEURON_CELL_POOL_HPP
#define NEURON_CELL_POOL_HPP

#include <thrust/device_vector.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class NeuronCellPool
{
    public:
        NeuronCellPool();
        virtual ~NeuronCellPool();

    protected:
        class CellDataPool
        {
            public:
                struct Synapse
                {
                    thrust::host_vector<int>   Link;
                    thrust::host_vector<float> Weight;
                };

                thrust::host_vector<Synapse> SynapseVector;
                thrust::host_vector<float>   Level;
                thrust::host_vector<float>   Output;
        };

        CellDataPool m_CellDataPool;
    private:
};

#endif // NEURON_CELL_POOL_HPP
