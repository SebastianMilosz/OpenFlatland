#ifndef NEURON_LAYER_VECTOR_HPP_INCLUDED
#define NEURON_LAYER_VECTOR_HPP_INCLUDED

#include <sigslot.h>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include "neuron_layer.hpp"

class NeuronLayerVector : public NeuronLayer
{
    CODEFRAME_META_CLASS_NAME( "NeuronLayerVector" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 NeuronLayerVector( const std::string& name, ObjectNode* parent, const std::string& link );
        virtual ~NeuronLayerVector() = default;

        codeframe::Property< thrust::host_vector<float> > Data;
        codeframe::Property<float> Max;
        codeframe::Property<float> Min;

        void ProcessData(thrust::host_vector<float>& vectData) override;

    private:
        float m_Max;
        float m_Min;

        struct copy_functor
        {
            public:
                copy_functor(thrust::host_vector<float>& vect, float& max, float& min) :
                    m_vect(vect),
                    m_Max(max),
                    m_Min(min)
                {
                }

                __device__ __host__ void operator()(float& refData)
                {
                    m_vect.push_back(refData);
                }

            private:
                thrust::host_vector<float>& m_vect;
                float& m_Max;
                float& m_Min;
        };
};

#endif // NEURON_LAYER_VECTOR_HPP_INCLUDED
