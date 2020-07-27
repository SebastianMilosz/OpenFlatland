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

        virtual void ProcessData(thrust::host_vector<float>& vectInData, thrust::host_vector<float>& vectOutData) = 0;
};

#endif // NEURON_LAYER_HPP_INCLUDED
