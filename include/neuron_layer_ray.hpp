#ifndef NEURON_LAYER_RAY_HPP_INCLUDED
#define NEURON_LAYER_RAY_HPP_INCLUDED

#include <sigslot.h>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include "entity_vision_node.hpp"
#include "neuron_layer.hpp"

class NeuronLayerRay : public NeuronLayer
{
    CODEFRAME_META_CLASS_NAME( "NeuronLayerRay" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        enum class Layer
        {
            LAYER_DISTANCE = 0U,
            LAYER_FIXTURE = 1U
        };

                 NeuronLayerRay( const std::string& name, ObjectNode* parent, const std::string& link );
        virtual ~NeuronLayerRay() = default;

        codeframe::Property< thrust::host_vector<RayData> > Data;
        codeframe::Property<float> MaxDistance;
        codeframe::Property<float> MinDistance;
        codeframe::Property<float> MaxFixture;
        codeframe::Property<float> MinFixture;

        void GiveData(thrust::host_vector<float>& vectData) override;
        void TakeData(thrust::host_vector<float>& vectData) override;

    private:
        float m_MaxDistance;
        float m_MinDistance;
        float m_MaxFixture;
        float m_MinFixture;

        template<Layer LAYER>
        struct copy_functor
        {
            public:
                copy_functor(thrust::host_vector<float>& vect, float& max, float& min) :
                    m_vect(vect),
                    m_Max(max),
                    m_Min(min)
                {
                }

                __device__ __host__ void operator()(RayData& refData);

            private:
                thrust::host_vector<float>& m_vect;
                float& m_Max;
                float& m_Min;
        };

        struct marge_functor
        {
            public:
                marge_functor(thrust::host_vector<float>& vect) :
                    m_vect(vect)
                    {
                    }

                template <typename Tuple>
                __device__ __host__ void operator()(Tuple& value)
                {
                    m_vect.push_back(thrust::get<1>(value));
                    m_vect.push_back(thrust::get<2>(value));
                }

            private:
                thrust::host_vector<float>& m_vect;
        };
};

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
__device__ __host__ inline void NeuronLayerRay::copy_functor<NeuronLayerRay::Layer::LAYER_DISTANCE>::operator()(RayData& refData)
{
    m_Min = std::fmin(m_Min,refData.Distance);
    m_Max = std::fmax(m_Max,refData.Distance);

    m_vect.push_back(refData.Distance);
};

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
template<>
__device__ __host__ inline void NeuronLayerRay::copy_functor<NeuronLayerRay::Layer::LAYER_FIXTURE>::operator()(RayData& refData)
{
    m_Min = std::fmin(m_Min,refData.Fixture);
    m_Max = std::fmax(m_Max,refData.Fixture);

    m_vect.push_back(refData.Fixture);
};

#endif // NEURON_LAYER_RAY_HPP_INCLUDED
