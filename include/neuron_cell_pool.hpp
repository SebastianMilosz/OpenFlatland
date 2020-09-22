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
        codeframe::Property< thrust::host_vector<float> > SynapseLink;
        codeframe::Property< thrust::host_vector<float> > SynapseWeight;
        codeframe::Property< thrust::host_vector<float> > IntegrateThreshold;
        codeframe::Property< thrust::host_vector<float> > IntegrateLevel;

        struct SynapseVector
        {
            thrust::host_vector<float> Link;
            thrust::host_vector<float> Weight;
            uint32_t                   Size = 0U;
        };

        NeuronCellPool( const std::string& name, ObjectNode* parent );
        virtual ~NeuronCellPool();

        void OnNeuronSynapseLimit(codeframe::PropertyNode* prop);
        void OnNeuronOutputLimit(codeframe::PropertyNode* prop);

        void Initialize(const codeframe::Point2D<unsigned int>& poolSize);
        void Calculate();
        void Populate();

    private:
        template<typename T>
        struct copy_range_functor
        {
            public:
                copy_range_functor(thrust::host_vector<T>& targetVector, const uint32_t targetSize) :
                    m_targetVector(targetVector),
                    m_targetSize(targetSize),
                    m_currentTargetPos(0U)
                {
                }

                __device__ __host__ void operator()(T value)
                {
                    if (m_currentTargetPos < m_targetSize)
                    {
                        m_targetVector[m_currentTargetPos++] = value;
                    }
                }

            private:
                thrust::host_vector<T>& m_targetVector;
                const uint32_t m_targetSize;
                uint32_t m_currentTargetPos;
        };

        struct neuron_calculate_functor
        {
            public:
                neuron_calculate_functor(const thrust::host_vector<uint64_t>& outputConsumedVector, const SynapseVector& synapseConsumedVector) :
                    m_outputConsumedVector(outputConsumedVector),
                    m_synapseConsumedVector(synapseConsumedVector)
                {
                }

                template <typename Tuple>
                __device__ __host__ void operator()(Tuple& value)
                {
                    volatile uint32_t n = thrust::get<0>(value);
                    volatile double link = m_synapseConsumedVector.Link[n];
                    volatile double weight = m_synapseConsumedVector.Weight[n];

                    if (link > 0.0d)
                    {
                        double intpart;
                        double fractpart = modf(link , &intpart);
                        uint64_t outVal = m_outputConsumedVector[intpart];

                    }
                }

            private:
                const thrust::host_vector<uint64_t>& m_outputConsumedVector;
                const SynapseVector&                 m_synapseConsumedVector;
        };

        constexpr static uint8_t MAX_SYNAPSE_CNT = 100U;

        SynapseVector                 m_Synapse;
        thrust::host_vector<float>    m_IntegrateLevel;
        thrust::host_vector<float>    m_IntegrateThreshold;
        thrust::host_vector<float>    m_IntegrateInterval;
        thrust::host_vector<uint64_t> m_Output;

        codeframe::Point2D<unsigned int> m_CurrentSize = codeframe::Point2D<unsigned int>(0U,0U);

        std::mt19937 m_generator;
        std::exponential_distribution<> m_distribution;
};

#endif // NEURON_CELL_POOL_HPP
