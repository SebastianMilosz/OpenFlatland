#ifndef SERIALIZABLENEURONLAYERCONTAINER_HPP_INCLUDED
#define SERIALIZABLENEURONLAYERCONTAINER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable.hpp>
#include <serializablecontainer.hpp>
#include <serializableinterface.hpp>

#include "serializableneuronlayer.hpp"

class SerializableNeuronLayerContainer : public codeframe::cSerializableContainer
{
        CODEFRAME_META_CLASS_NAME( "SerializableNeuronLayerContainer" );
        CODEFRAME_META_BUILD_ROLE( codeframe::CONTAINER );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        SerializableNeuronLayerContainer( std::string name, cSerializableInterface* parent );
        virtual ~SerializableNeuronLayerContainer();

        void Calculate();

        codeframe::Property<unsigned int, SerializableNeuronLayerContainer> LayersCnt;

    protected:
        smart_ptr<codeframe::cSerializableInterface> Create(
                                                             const std::string& className,
                                                             const std::string& objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                            );

    void SetLayersCnt( unsigned int cnt );
};

#endif // SERIALIZABLENEURONLAYERCONTAINER_HPP_INCLUDED
