#ifndef SERIALIZABLENEURONLAYERCONTAINER_HPP_INCLUDED
#define SERIALIZABLENEURONLAYERCONTAINER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

#include "neuron_layer.hpp"

class SerializableNeuronLayerContainer : public codeframe::ObjectContainer
{
        CODEFRAME_META_CLASS_NAME( "NeuronLayerContainer" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 SerializableNeuronLayerContainer( const std::string& name, ObjectNode* parent );
        virtual ~SerializableNeuronLayerContainer() = default;

        void Calculate();

        codeframe::Property<unsigned int> LayersCnt;

    protected:
        smart_ptr<codeframe::ObjectSelection> Create(
                                                 const std::string& className,
                                                 const std::string& objName,
                                                 const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                );

    void SetLayersCnt( unsigned int cnt );
};

#endif // SERIALIZABLENEURONLAYERCONTAINER_HPP_INCLUDED
