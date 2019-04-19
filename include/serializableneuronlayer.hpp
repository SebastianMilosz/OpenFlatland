#ifndef SERIALIZABLENEURONLAYER_HPP_INCLUDED
#define SERIALIZABLENEURONLAYER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable.hpp>
#include <serializablecontainer.hpp>
#include <serializableinterface.hpp>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

class SerializableNeuronLayer : public codeframe::cSerializable
{
        CODEFRAME_META_CLASS_NAME( "SerializableNeuronLayer" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        SerializableNeuronLayer( const std::string& name, cSerializableInterface* parent );
        virtual ~SerializableNeuronLayer();

        void Calculate();

        codeframe::Property< unsigned int > Activation;
        codeframe::Property< std::vector<unsigned int>, SerializableNeuronLayer > WeightDimensions;
        codeframe::Property< thrust::host_vector<float> > WeightMatrix;
        codeframe::Property< thrust::host_vector<float> > Input;
        codeframe::Property< thrust::host_vector<float> > Output;

    protected:
        bool InitializeNetwork();

    private:
        void OnWeightDimensionsVectorChanged( codeframe::PropertyBase* prop );
        const std::vector<unsigned int>& GetWeightDimensionsVector();

        std::vector<unsigned int> m_WeightDimensions;
};

#endif // SERIALIZABLENEURONLAYER_HPP_INCLUDED
