#ifndef SERIALIZABLENEURONLAYER_HPP_INCLUDED
#define SERIALIZABLENEURONLAYER_HPP_INCLUDED

#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

class SerializableNeuronLayer : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME( "SerializableNeuronLayer" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        SerializableNeuronLayer( const std::string& name, ObjectNode* parent );
        virtual ~SerializableNeuronLayer();

        void Calculate();

        codeframe::Property< unsigned int > Activation;
        codeframe::Property< std::vector<unsigned int> > WeightDimensions;
        codeframe::Property< thrust::host_vector<float> > WeightMatrix;
        codeframe::Property< thrust::host_vector<float> > Input;
        codeframe::Property< thrust::host_vector<float> > Output;

    protected:
        bool InitializeNetwork();

    private:
        void OnWeightDimensionsVectorChanged( codeframe::PropertyBase* prop );
        const std::vector<unsigned int>& GetWeightDimensionsVector();

        std::vector<unsigned int> m_WeightDimensions;

        // Calculation internal vectors
        thrust::host_vector<float> m_WeightVector;
        thrust::host_vector<float> m_MovingInputLayerVector;
        thrust::host_vector<float> m_MovingOutputLayerVector;

};

#endif // SERIALIZABLENEURONLAYER_HPP_INCLUDED
