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

        void ProcessData(thrust::host_vector<float>& vectData) override;

    private:
        struct normalize_functor
        {
            public:
                normalize_functor(thrust::host_vector<float>& vect) :
                    m_vect(vect)
                {
                }

                __device__ __host__ void operator()(RayData& refData)
                {
                    m_vect.push_back(refData.Distance);
                    m_vect.push_back(refData.Fixture);
                }

            private:
                thrust::host_vector<float>& m_vect;
        };
};

#endif // NEURON_LAYER_RAY_HPP_INCLUDED
