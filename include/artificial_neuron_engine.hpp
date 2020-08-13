#ifndef ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED
#define ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED

#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>

#include "neuron_layer_container.hpp"

class ArtificialNeuronEngine : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME( "ArtificialNeuronEngine" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 ArtificialNeuronEngine( const std::string& name, ObjectNode* parent );
        virtual ~ArtificialNeuronEngine() = default;

        codeframe::Property< thrust::host_vector<float> > Input;
        codeframe::Property< thrust::host_vector<float> > Output;

        void Calculate();

        void OnWeightDimensionsVectorChanged( codeframe::PropertyNode* prop );

    protected:
        NeuronLayerContainer m_Inputs;
        NeuronLayerContainer m_Outputs;

        virtual void CollectInputs();
        virtual void ProcesseOutputs();
    private:
        thrust::host_vector<float> m_vectInData;
        thrust::host_vector<float> m_vectOutData;
};

#endif // ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED
