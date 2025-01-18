#include "neuron_layer_container.hpp"

#include "neuron_layer_ray.hpp"
#include "neuron_layer_vector.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
NeuronLayerContainer::NeuronLayerContainer( const std::string& name, ObjectNode* parent ) :
    ObjectContainer( name, parent )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<codeframe::Object> NeuronLayerContainer::Create(
                                                     const std::string& className,
                                                     const std::string& objName,
                                                     const std::vector<codeframe::VariantValue>& params
                                                    )
{
    std::string link("");

    for ( auto it = params.begin(); it != params.end(); ++it )
    {
        if ( it->GetType() == codeframe::TYPE_TEXT )
        {
                 if ( it->IsName( "href" ) )
            {
                link = it->ValueString;
            }
        }
    }

    if ( className == "NeuronLayerVector" )
    {
        auto obj = smart_ptr<NeuronLayer>( new NeuronLayerVector( objName, this, link ) );

        (void)InsertObject( obj );

        return obj;
    }
    else if ( className == "NeuronLayerRay" )
    {
        auto obj = smart_ptr<NeuronLayer>( new NeuronLayerRay( objName, this, link ) );

        (void)InsertObject( obj );

        return obj;
    }

    return smart_ptr<codeframe::Object>();
}
