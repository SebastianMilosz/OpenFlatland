#ifndef NEURON_CELL_POOL_HPP
#define NEURON_CELL_POOL_HPP

#include <random>
#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>

#include <thrust/device_vector.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class NeuronCellPool : public codeframe::Object
{
    CODEFRAME_META_CLASS_NAME( "NeuronCellPool" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        codeframe::Property< unsigned int > NeuronSynapseLimit;
        codeframe::Property< unsigned int > NeuronOutputLimit;

        // Those vectors are used to store current neuron pool state in nvs
        codeframe::Property< thrust::host_vector<uint32_t> > SynapseLink;
        codeframe::Property< thrust::host_vector<float> >    SynapseWeight;
        codeframe::Property< thrust::host_vector<float> >    IntegrateThreshold;
        codeframe::Property< thrust::host_vector<float> >    IntegrateLevel;

        struct SynapseVector
        {
            thrust::host_vector<uint32_t> Link;
            thrust::host_vector<float>    Weight;
        };

        NeuronCellPool( const std::string& name, ObjectNode* parent );
        virtual ~NeuronCellPool();

        void Initialize(const uint32_t cnt);
        void Calculate();
        void Populate();

    private:
        constexpr static uint8_t MAX_SYNAPSE_CNT = 100U;
        constexpr static uint8_t MAX_OUTPUT_CNT = 100U;

        SynapseVector              m_Synapse;
        thrust::host_vector<float> m_IntegrateLevel;
        thrust::host_vector<float> m_IntegrateThreshold;
        thrust::host_vector<bool>  m_Output;

        uint32_t m_CurrentSize = 0U;
        uint32_t m_CurrentSynapseLimit = 0U;
        uint32_t m_CurrentOutputLimit = 0U;

        std::mt19937 m_generator;
        std::exponential_distribution<> m_distribution;
};

#endif // NEURON_CELL_POOL_HPP
