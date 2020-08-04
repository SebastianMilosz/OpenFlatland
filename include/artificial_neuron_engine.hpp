#ifndef ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED
#define ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED

#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

#include "neuron_layer.hpp"

class ArtificialNeuronEngine : public codeframe::ObjectContainer
{
        CODEFRAME_META_CLASS_NAME( "ArtificialNeuronEngine" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 ArtificialNeuronEngine( const std::string& name, ObjectNode* parent );
        virtual ~ArtificialNeuronEngine() = default;

        void Calculate();

        void OnWeightDimensionsVectorChanged( codeframe::PropertyNode* prop );

    protected:
        smart_ptr<codeframe::ObjectSelection> Create(
                                                 const std::string& className,
                                                 const std::string& objName,
                                                 const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                );

    private:
        thrust::host_vector<float> m_vectInData;
        thrust::host_vector<float> m_vectOutData;
};

#endif // ARTIFICIAL_NEURON_ENGINE_HPP_INCLUDED
