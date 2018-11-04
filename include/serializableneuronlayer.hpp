#ifndef SERIALIZABLENEURONLAYER_HPP_INCLUDED
#define SERIALIZABLENEURONLAYER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable.hpp>
#include <serializablecontainer.hpp>
#include <serializableinterface.hpp>

class SerializableNeuronLayer : public codeframe::cSerializable
{
        CODEFRAME_META_CLASS_NAME( "SerializableNeuronLayer" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        SerializableNeuronLayer( std::string name, cSerializableInterface* parent );
        virtual ~SerializableNeuronLayer();

        void Calculate();

        codeframe::Property< unsigned int > Activation;
        codeframe::Property< std::vector<unsigned int>, SerializableNeuronLayer > WeightDimensions;
        codeframe::Property< std::vector<float> > WeightMatrix;
        codeframe::Property< std::vector<float> > Input;
        codeframe::Property< std::vector<float> > Output;

    protected:
        void SetNeuronCnt( std::vector<unsigned int> cntVec );
};

#endif // SERIALIZABLENEURONLAYER_HPP_INCLUDED
