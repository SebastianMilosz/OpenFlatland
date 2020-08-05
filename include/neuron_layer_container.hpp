#ifndef NEURON_LAYER_CONTAINER_HPP_INCLUDED
#define NEURON_LAYER_CONTAINER_HPP_INCLUDED

#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

#include "neuron_layer.hpp"

class NeuronLayerContainer : public codeframe::ObjectContainer
{
        CODEFRAME_META_CLASS_NAME( "NeuronLayerContainer" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 NeuronLayerContainer( const std::string& name, ObjectNode* parent );
        virtual ~NeuronLayerContainer() = default;

        smart_ptr<codeframe::ObjectSelection> Create(
                                                 const std::string& className,
                                                 const std::string& objName,
                                                 const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                );
};

#endif // NEURON_LAYER_CONTAINER_HPP_INCLUDED
