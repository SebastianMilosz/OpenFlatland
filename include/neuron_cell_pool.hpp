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

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class NeuronCellPool : public NeuronModel::Column::Model_SNN
{
    CODEFRAME_META_CLASS_NAME( "NeuronCellPool" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        codeframe::Property< codeframe::Point2D<unsigned int> > CellPoolSize;
        codeframe::Property< unsigned int >                     NeuronSynapseLimit;
        codeframe::Property< unsigned int >                     NeuronOutputLimit;
        codeframe::Property< thrust::host_vector<float> >       SynapseLink;
        codeframe::Property< thrust::host_vector<float> >       SynapseWeight;
        codeframe::Property< thrust::host_vector<float> >       IntegrateThreshold;
        codeframe::Property< thrust::host_vector<float> >       IntegrateLevel;

                 NeuronCellPool(const std::string& name, ObjectNode* parent);
        virtual ~NeuronCellPool();

        void Calculate(const thrust::host_vector<float>& dataInput, thrust::host_vector<float>& dataOutput) override;

        uint32_t GetSynapseSize() { return m_Synapse.Size; }

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
        void OnNeuronSynapseLimit(codeframe::PropertyNode* prop);
        void OnNeuronOutputLimit(codeframe::PropertyNode* prop);
        void OnCellPoolSize(codeframe::PropertyNode* prop);

        void Initialize(unsigned int w, unsigned int h);
        void Resize(uint32_t width, uint32_t height);
        void Populate(const thrust::host_vector<float>& dataInput, thrust::host_vector<float>& dataOutput);

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

        constexpr static uint8_t MAX_SYNAPSE_CNT = 100U;

        SynapseVector                 m_Synapse;
        thrust::host_vector<float>    m_IntegrateLevel;
        thrust::host_vector<float>    m_IntegrateThreshold;
        thrust::host_vector<float>    m_IntegrateInterval;
        thrust::host_vector<uint64_t> m_Output;
        std::mt19937                  m_generator;
        uint32_t                      m_populateDelay;

        codeframe::Point2D<unsigned int> m_CurrentSize = codeframe::Point2D<unsigned int>(0U,0U);
};

#endif // NEURON_CELL_POOL_HPP
