#ifndef SERIALIZABLENEURONLAYER_HPP_INCLUDED
#define SERIALIZABLENEURONLAYER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable.hpp>
#include <serializablecontainer.hpp>
#include <serializableinterface.hpp>

#include "serializableneuron.hpp"

class SerializableNeuronLayer : public codeframe::cSerializableContainer
{
    public:
        std::string Role()            const { return "Container";               }
        std::string Class()           const { return "SerializableNeuronLayer"; }
        std::string BuildType()       const { return "Dynamic";                 }
        std::string ConstructPatern() const { return ""; }

    public:
        SerializableNeuronLayer( std::string name, cSerializableInterface* parent );
        virtual ~SerializableNeuronLayer();

        void Calculate();

        codeframe::Property< unsigned int, SerializableNeuronLayer > NeuronCnt;
        codeframe::Property< std::vector<float> > Input;
        codeframe::Property< std::vector<float> > Output;

    protected:
        smart_ptr<codeframe::cSerializableInterface> Create(
                                                             const std::string& className,
                                                             const std::string& objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                            );

        void SetNeuronCnt( unsigned int cnt );
};

#endif // SERIALIZABLENEURONLAYER_HPP_INCLUDED
