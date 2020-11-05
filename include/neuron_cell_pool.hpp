#ifndef NEURON_CELL_POOL_HPP
#define NEURON_CELL_POOL_HPP

#include <random>
#include <chrono>
#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <thrust/random.h>

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

        NeuronCellPool( const std::string& name, ObjectNode* parent,
                        const thrust::host_vector<float>& inData,
                              thrust::host_vector<float>& outData );
        virtual ~NeuronCellPool();

        uint32_t GetSynapseSize() { return m_Synapse.Size; }

        void OnNeuronSynapseLimit(codeframe::PropertyNode* prop);
        void OnNeuronOutputLimit(codeframe::PropertyNode* prop);

        void Initialize(const codeframe::Point2D<unsigned int>& poolSize);
        void Calculate();
        void Populate();

        uint32_t CoordinateToOffset(const uint32_t x, const uint32_t y) const
        {
            return m_CurrentSize.Y() * y + x;
        }

        codeframe::Point2D<unsigned int> OffsetToCoordinate(const uint32_t offset) const
        {
            codeframe::Point2D<unsigned int> retValue;
            retValue.SetX(offset % m_CurrentSize.X());
            retValue.SetY(std::floor(offset / m_CurrentSize.X()));
            return retValue;
        }

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
                neuron_calculate_functor(const thrust::host_vector<uint64_t>& outputConsumedVector,
                                               SynapseVector& synapseConsumedVector,
                                         const thrust::host_vector<float>& inData) :
                    m_outputConsumedVector(outputConsumedVector),
                    m_synapseConsumedVector(synapseConsumedVector),
                    m_inData(inData),
                    m_inDataSize(m_inData.size())
                {
                }

                template <typename Tuple>
                __device__ __host__ void operator()(Tuple& value)
                {
                    uint32_t n = thrust::get<0>(value);
                    uint32_t s = m_synapseConsumedVector.Size;

                    for (uint32_t i = 0U; i < s; i++)
                    {
                        volatile double link = m_synapseConsumedVector.Link[n * s + i];

                        // Link to internal pool space
                        if (link > 0.0d)
                        {
                            double intpart;
                            uint8_t bitPos = 64U * modf(link , &intpart);

                            if (intpart < m_outputConsumedVector.size())
                            {
                                uint64_t outVal = m_outputConsumedVector[intpart];
                                double weight = m_synapseConsumedVector.Weight[n * s + i];
                                thrust::get<1>(value) += (outVal & (1U<<bitPos)) * weight;
                            }
                        }
                        // Link to external inputs space
                        else if (link < 0.0d)
                        {
                            volatile unsigned int inPos = std::fabs(link);
                            if (inPos < m_inDataSize)
                            {
                                volatile double weight = m_synapseConsumedVector.Weight[n * s + i];
                                volatile double inValue = m_inData[inPos];

                                volatile double newValue = thrust::get<1>(value) + inValue * weight;
                                thrust::get<1>(value) = newValue;

                                if (newValue > 0)
                                {
                                    m_synapseConsumedVector.Weight[n * s + i] += 0.1f;
                                }
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                }

            private:
                const thrust::host_vector<uint64_t>& m_outputConsumedVector;
                      SynapseVector&                 m_synapseConsumedVector;
                const thrust::host_vector<float>&    m_inData;
                const unsigned int                   m_inDataSize;
        };

        struct neuron_output_calculate_functor
        {
            public:
                constexpr static uint8_t TUPLE_POS_INTEGRATE_LEVEL = 0U;
                constexpr static uint8_t TUPLE_POS_INTEGRATE_THRESHOLD = 1U;
                constexpr static uint8_t TUPLE_POS_INTEGRATE_INTERVAL = 2U;
                constexpr static uint8_t TUPLE_POS_INTEGRATE_OUTPUT = 3U;

                neuron_output_calculate_functor()
                {
                }

                template <typename Tuple>
                __device__ __host__ void operator()(Tuple& value)
                {
                    thrust::get<TUPLE_POS_INTEGRATE_OUTPUT>(value) = thrust::get<TUPLE_POS_INTEGRATE_OUTPUT>(value) << 1U;

                    if (thrust::get<TUPLE_POS_INTEGRATE_LEVEL>(value) > thrust::get<TUPLE_POS_INTEGRATE_THRESHOLD>(value))
                    {
                        thrust::get<TUPLE_POS_INTEGRATE_INTERVAL>(value)++;
                        thrust::get<TUPLE_POS_INTEGRATE_LEVEL>(value) = -10.0f; // Hyperpolaryzation begin
                        thrust::get<TUPLE_POS_INTEGRATE_OUTPUT>(value) &= static_cast<uint64_t>(0x01U);
                    }
                    else
                    {
                        if (thrust::get<TUPLE_POS_INTEGRATE_LEVEL>(value) > 0.0f)
                        {
                            // Normal charge pump positive depolarization
                            thrust::get<TUPLE_POS_INTEGRATE_LEVEL>(value) -= 0.0001;
                        }
                        else if (thrust::get<TUPLE_POS_INTEGRATE_LEVEL>(value) < 0.0f) // Hyperpolaryzation
                        {
                            // Normal charge pump negative depolarization
                            thrust::get<TUPLE_POS_INTEGRATE_LEVEL>(value) += 1.0f;
                        }
                    }
                }
        };

        template<class T>
        struct neuron_output_take_functor
        {
            public:
                neuron_output_take_functor(thrust::host_vector<float>& integrateLevelVector) :
                    m_countValue(0U),
                    m_integrateLevelVector(integrateLevelVector)
                {
                }

                __device__ __host__ T operator()(T& value)
                {
                    volatile float newValue = m_integrateLevelVector[m_integrateLevelVector.size() - 1U - m_countValue++];
                    return newValue;
                }
            private:
                uint32_t m_countValue;
                thrust::host_vector<float>& m_integrateLevelVector;
        };

        struct neuron_populate_functor
        {
            public:
                neuron_populate_functor(const thrust::host_vector<uint64_t>& outputConsumedVector,
                                              SynapseVector& synapseConsumedVector,
                                        const thrust::host_vector<float>& inData,
                                        const codeframe::Point2D<unsigned int>& poolSize,
                                        std::mt19937& generator) :
                    m_outputConsumedVector(outputConsumedVector),
                    m_synapseConsumedVector(synapseConsumedVector),
                    m_inData(inData),
                    m_inDataSize(m_inData.size()),
                    m_poolSizeWidth(poolSize.X()),
                    m_poolSizeHeight(poolSize.Y()),
                    m_distributionReal_1(-2.0f, 2.0f),
                    m_distributionReal_2(0.0f, 1.0f),
                    m_distributionReal_3((-1.0f) * (float)m_inDataSize, 0.0f),
                    m_generator(generator)

                {
                    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
                    m_generator.seed(seed);
                }

                template <typename Tuple>
                __device__ __host__ void operator()(Tuple& value)
                {
                    uint32_t n = thrust::get<0>(value);
                    uint32_t s = m_synapseConsumedVector.Size;

                    for (uint32_t i = 0U; i < s; i++)
                    {
                        if (m_synapseConsumedVector.Link[n * s + i] == 0.0 ||
                            m_synapseConsumedVector.Weight[n * s + i] < 0.02)
                        {
                            m_generator.discard(m_generator.state_size);

                            volatile float nodeLinkRandomX = (float)OffsetToCoordinate(n).X() + (float)m_distributionReal_1(m_generator);
                            volatile float nodeLinkRandomY = (float)OffsetToCoordinate(n).Y() + (float)m_distributionReal_1(m_generator);

                            volatile float targetLink = CoordinateToOffset((int)nodeLinkRandomX, (int)nodeLinkRandomY) + m_distributionReal_2(m_generator);

                            if (targetLink < 0.0f)
                            {
                                targetLink = m_distributionReal_3(m_generator);
                            }

                            m_synapseConsumedVector.Link[n * s + i] = targetLink;
                            m_synapseConsumedVector.Weight[n * s + i] = 0.01;
                            break;
                        }
                    }
                }

                float CoordinateToOffset(const float x, const float y) const
                {
                    return (float)m_poolSizeWidth * y + x;
                }

                int CoordinateToOffset(const int x, const int y) const
                {
                    return m_poolSizeWidth * y + x;
                }

                codeframe::Point2D<unsigned int> OffsetToCoordinate(const uint32_t offset) const
                {
                    codeframe::Point2D<unsigned int> retValue;
                    retValue.SetX(offset % m_poolSizeWidth);
                    retValue.SetY(std::floor(offset / m_poolSizeWidth));
                    return retValue;
                }

                codeframe::Point2D<int> OffsetToCoordinate(const int32_t offset) const
                {
                    codeframe::Point2D<int> retValue;
                    retValue.SetX(offset % m_poolSizeWidth);
                    retValue.SetY(std::floor(offset / m_poolSizeWidth));
                    return retValue;
                }

            private:
                const thrust::host_vector<uint64_t>&  m_outputConsumedVector;
                      SynapseVector&                  m_synapseConsumedVector;
                const thrust::host_vector<float>&     m_inData;
                const unsigned int                    m_inDataSize;
                const unsigned int                    m_poolSizeWidth;
                const unsigned int                    m_poolSizeHeight;
                std::uniform_real_distribution<float> m_distributionReal_1;
                std::uniform_real_distribution<float> m_distributionReal_2;
                std::uniform_real_distribution<float> m_distributionReal_3;
                std::mt19937&                         m_generator;
        };

        constexpr static uint8_t MAX_SYNAPSE_CNT = 100U;

        SynapseVector                 m_Synapse;
        thrust::host_vector<float>    m_IntegrateLevel;
        thrust::host_vector<float>    m_IntegrateThreshold;
        thrust::host_vector<float>    m_IntegrateInterval;
        thrust::host_vector<uint64_t> m_Output;
        std::mt19937                  m_generator;

        const thrust::host_vector<float>& m_vectInData;
              thrust::host_vector<float>& m_vectOutData;

        codeframe::Point2D<unsigned int> m_CurrentSize = codeframe::Point2D<unsigned int>(0U,0U);
};

#endif // NEURON_CELL_POOL_HPP
