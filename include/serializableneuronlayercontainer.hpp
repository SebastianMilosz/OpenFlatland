#ifndef SERIALIZABLENEURONLAYERCONTAINER_HPP_INCLUDED
#define SERIALIZABLENEURONLAYERCONTAINER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable.hpp>
#include <serializablecontainer.hpp>
#include <serializableinterface.hpp>

#include "serializableneuronlayer.hpp"

class SerializableNeuronLayerContainer : public codeframe::cSerializableContainer
{
    public:
        std::string Role()            const { return "Container";                        }
        std::string Class()           const { return "SerializableNeuronLayerContainer"; }
        std::string BuildType()       const { return "Static";                           }
        std::string ConstructPatern() const { return ""; }

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
