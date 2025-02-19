#ifndef NEURON_LAYER_HPP_INCLUDED
#define NEURON_LAYER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include "entity_vision_node.hpp"

class NeuronLayer : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME( "NeuronLayer" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
                 NeuronLayer( const std::string& name, ObjectNode* parent, const std::string& link );
        virtual ~NeuronLayer() = default;

        virtual uint32_t size() const = 0;
        virtual void GiveData(thrust::host_vector<float>& vectData) = 0;
        virtual uint32_t TakeData(thrust::host_vector<float>& vectData, uint32_t vectPos) = 0;

    protected:
        struct normalize_functor
        {
            public:
                normalize_functor(float& max, float& min) :
                    m_Max(max),
                    m_Min(min)
                {
                }

                __device__ __host__ void operator()(float& refData)
                {
                    refData = (refData - m_Min) / (m_Max-m_Min);
                }

            private:
                float& m_Max;
                float& m_Min;
        };
};

#endif // NEURON_LAYER_HPP_INCLUDED
