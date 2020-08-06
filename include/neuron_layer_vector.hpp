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
        codeframe::Property<float> Evg;

        void ProcessData(thrust::host_vector<float>& vectData) override;

    private:
        float m_Max;
        float m_Min;
        float m_Evg;

        struct normalize_functor
        {
            public:
                normalize_functor(thrust::host_vector<float>& vect) :
                    m_vect(vect)
                {
                }

                __device__ __host__ void operator()(float& refData)
                {
                    m_vect.push_back(refData);
                }

            private:
                thrust::host_vector<float>& m_vect;
        };
};

#endif // NEURON_LAYER_VECTOR_HPP_INCLUDED
