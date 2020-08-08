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
                 NeuronLayerRay( const std::string& name, ObjectNode* parent, const std::string& link );
        virtual ~NeuronLayerRay() = default;

        codeframe::Property< thrust::host_vector<RayData> > Data;
        codeframe::Property<float> MaxDistance;
        codeframe::Property<float> MinDistance;
        codeframe::Property<float> MaxFixture;
        codeframe::Property<float> MinFixture;

        void ProcessData(thrust::host_vector<float>& vectData) override;

    private:
        float m_MaxDistance;
        float m_MinDistance;
        float m_MaxFixture;
        float m_MinFixture;

        struct copy_functor
        {
            public:
                copy_functor(thrust::host_vector<float>& vect, unsigned int layer, float& max, float& min) :
                    m_vect(vect),
                    m_Max(max),
                    m_Min(min),
                    m_layer(layer)
                {
                }

                __device__ __host__ void operator()(RayData& refData)
                {
                    float value = 0.0f;

                    if (m_layer == 0U)
                    {
                        value = refData.Distance;
                    }
                    else
                    {
                        value = refData.Fixture;
                    }

                    if (value > m_Max)
                    {
                        m_Max = value;
                    }
                    else if (value < m_Min)
                    {
                        m_Min = value;
                    }
                    m_vect.push_back(value);
                }

            private:
                thrust::host_vector<float>& m_vect;
                float& m_Max;
                float& m_Min;
                unsigned int m_layer;
        };
};

#endif // NEURON_LAYER_RAY_HPP_INCLUDED
