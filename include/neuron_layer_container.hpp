#ifndef NEURON_LAYER_CONTAINER_HPP_INCLUDED
#define NEURON_LAYER_CONTAINER_HPP_INCLUDED

#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

#include "neuron_layer.hpp"

class NeuronLayerContainer : public codeframe::ObjectContainer
{
        CODEFRAME_META_CLASS_NAME( "NeuronLayerContainer" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 NeuronLayerContainer( const std::string& name, ObjectNode* parent );
        virtual ~NeuronLayerContainer() = default;

        const std::vector< std::string >& ClassSet() const override
        {
            return m_classSet;
        }

        const std::vector< std::vector<std::string> >& ClassParameterSet() const override
        {
            return m_classParameterSet;
        }

        smart_ptr<codeframe::Object> Create(
                                             const std::string& className,
                                             const std::string& objName,
                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                           );

    private:
        const std::vector< std::string > m_classSet = {"NeuronLayerVector", "NeuronLayerRay"};
        const std::vector< std::vector<std::string> > m_classParameterSet;
};

#endif // NEURON_LAYER_CONTAINER_HPP_INCLUDED
