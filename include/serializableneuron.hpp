#ifndef SERIALIZABLENEURON_HPP_INCLUDED
#define SERIALIZABLENEURON_HPP_INCLUDED

#include <serializable.hpp>

class SerializableNeuron : public codeframe::cSerializable
{
        CODEFRAME_META_CLASS_NAME( "SerializableNeuron" );
        CODEFRAME_META_BUILD_ROLE( codeframe::OBJECT );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        SerializableNeuron(
                            std::string name,
                            cSerializableInterface* parent,
                            unsigned int inputCnt
                          );
        virtual ~SerializableNeuron();

        void Calculate();

        codeframe::Property< std::vector<float> > InputsWeights;
        codeframe::Property< float >              Output;
};

#endif // SERIALIZABLENEURON_HPP_INCLUDED
