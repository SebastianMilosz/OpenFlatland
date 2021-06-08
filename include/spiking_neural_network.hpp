#ifndef NEURON_CELL_POOL_HPP
#define NEURON_CELL_POOL_HPP

#include <random>
#include <chrono>
#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <thrust/random.h>

#include <thrust/device_vector.h>

#include "neuron_column_model_s1.hpp"
#include "copy_range_functor.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class SpikingNeuralNetwork : public NeuronModel::Column::Model_SNN
{
    CODEFRAME_META_CLASS_NAME( "SpikingNeuralNetwork" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        codeframe::Property< codeframe::Point2D<unsigned int> > CellPoolSize;
        codeframe::Property< unsigned int >                     NeuronSynapseLimit;
        codeframe::Property< unsigned int >                     NeuronOutputLimit;
        codeframe::Property< thrust::host_vector<float> >       SynapseLink;
        codeframe::Property< thrust::host_vector<float> >       SynapseWeight;
        codeframe::Property< thrust::host_vector<float> >       IntegrateThreshold;
        codeframe::Property< thrust::host_vector<float> >       IntegrateLevel;

                 SpikingNeuralNetwork(const std::string& name, ObjectNode* parent);
        virtual ~SpikingNeuralNetwork();

        void Calculate(const thrust::host_vector<float>& dataInput, thrust::host_vector<float>& dataOutput) override;

    protected:
        SynapseVector m_Synapse;
        thrust::host_vector<uint64_t> m_Output;
        thrust::host_vector<float>    m_IntegrateLevel;
        thrust::host_vector<float>    m_IntegrateThreshold;
        thrust::host_vector<float>    m_IntegrateInterval;
        codeframe::Point2D<unsigned int> m_CurrentSize = codeframe::Point2D<unsigned int>(0U,0U);

    private:
        void OnNeuronSynapseLimit(codeframe::PropertyNode* prop);
        void OnNeuronOutputLimit(codeframe::PropertyNode* prop);
        void OnCellPoolSize(codeframe::PropertyNode* prop);

        void Initialize(unsigned int w, unsigned int h);
        void Resize(uint32_t width, uint32_t height);
        void Populate(const thrust::host_vector<float>& dataInput, thrust::host_vector<float>& dataOutput);

        constexpr static uint8_t MAX_SYNAPSE_CNT = 100U;

        std::mt19937                  m_generator;
        uint32_t                      m_populateDelay;
};

#endif // NEURON_CELL_POOL_HPP
