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
        codeframe::Property<float> EvgDistance;
        codeframe::Property<float> MaxFixture;
        codeframe::Property<float> MinFixture;
        codeframe::Property<float> EvgFixture;

        void ProcessData(thrust::host_vector<float>& vectData) override;

    private:
        float m_MaxDistance;
        float m_MinDistance;
        float m_EvgDistance;
        float m_MaxFixture;
        float m_MinFixture;
        float m_EvgFixture;

        struct copy_functor
        {
            public:
                copy_functor(thrust::host_vector<float>& vect, float& max, float& min) :
                    m_vect(vect),
                    m_Max(max),
                    m_Min(min)
                {
                }

                __device__ __host__ void operator()(RayData& refData)
                {
                    if (refData.Distance > m_Max)
                    {
                        m_Max = refData.Distance;
                    }
                    else if (refData.Distance < m_Min)
                    {
                        m_Min = refData.Distance;
                    }

                    m_vect.push_back(refData.Distance);
                    m_vect.push_back(refData.Fixture);
                }

            private:
                thrust::host_vector<float>& m_vect;
                float& m_Max;
                float& m_Min;
        };
};

#endif // NEURON_LAYER_RAY_HPP_INCLUDED
