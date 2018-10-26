#ifndef SERIALIZABLENEURONLAYER_HPP_INCLUDED
#define SERIALIZABLENEURONLAYER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable.hpp>
#include <serializablecontainer.hpp>
#include <serializableinterface.hpp>

#include "serializableneuron.hpp"

class SerializableNeuronLayer : public codeframe::cSerializableContainer
{
        CODEFRAME_META_CLASS_NAME( "SerializableNeuronLayer" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

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
