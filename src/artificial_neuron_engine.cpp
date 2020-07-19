#include "artificial_neuron_engine.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ArtificialNeuronEngine::ArtificialNeuronEngine( const std::string& name, ObjectNode* parent ) :
    ObjectContainer( name, parent ),
    InterfaceLayersCnt( this, "InterfaceLayersCnt" , 2U , cPropertyInfo().Kind( KIND_NUMBER ).Description("InterfaceLayersCnt"), nullptr, std::bind(&ArtificialNeuronEngine::SetInterfaceLayersCnt, this, std::placeholders::_1) )
{
    SetInterfaceLayersCnt( (unsigned int)InterfaceLayersCnt );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::Calculate()
{
    for ( unsigned int n = 0U; n < Count(); n++ )
    {
        smart_ptr<ObjectNode> serializableObj = Get( n );

        NeuronLayer* neuronLayerObj = static_cast<NeuronLayer*>( smart_ptr_getRaw( serializableObj ) );

        if ( (NeuronLayer*)nullptr != neuronLayerObj )
        {
            neuronLayerObj->Calculate();
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<codeframe::ObjectSelection> ArtificialNeuronEngine::Create(
                                                     const std::string& className,
                                                     const std::string& objName,
                                                     const std::vector<codeframe::VariantValue>& params
                                                    )
{
    if ( className == "NeuronLayer" )
    {
        auto obj = smart_ptr<NeuronLayer>( new NeuronLayer( objName, NULL ) );

        (void)InsertObject( obj );

        return smart_ptr<codeframe::ObjectSelection>(new codeframe::ObjectSelection(obj));
    }

    return smart_ptr<codeframe::ObjectSelection>();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ArtificialNeuronEngine::SetInterfaceLayersCnt( unsigned int cnt )
{
    unsigned int thisCnt( Count() );
    // Set layer cnt to be at least configured
    if ( cnt > thisCnt )
    {
        unsigned int newCnt( cnt - thisCnt );
        CreateRange( "NeuronLayer", "InterfaceLayer", newCnt );
    }
}
