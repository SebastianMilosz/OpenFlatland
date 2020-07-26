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
    CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
                 NeuronLayerRay( const std::string& name, ObjectNode* parent, const std::string& link );
        virtual ~NeuronLayerRay() = default;

        codeframe::Property< thrust::host_vector<RayData> > Input;

        void Calculate() override;
};

#endif // NEURON_LAYER_RAY_HPP_INCLUDED
