#include "serializableneuronlayer.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayer::SerializableNeuronLayer( std::string name, cSerializableInterface* parent ) :
    cSerializableContainer( name, parent ),
    NeuronCnt( this, "NeuronCnt" , 10U , cPropertyInfo().Kind( KIND_NUMBER ).Description("NeuronCnt"), this, NULL, &SerializableNeuronLayer::SetNeuronCnt)
{
    SetNeuronCnt( (unsigned int)NeuronCnt );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
SerializableNeuronLayer::~SerializableNeuronLayer()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayer::Calculate()
{
    for ( unsigned int n = 0; n < Count(); n++ )
    {
        smart_ptr<cSerializableInterface> serializableObj = Get( n );

        SerializableNeuron* neuronObj = static_cast<SerializableNeuron*>( smart_ptr_getRaw( serializableObj ) );

        if ( (SerializableNeuron*)NULL != neuronObj )
        {
            neuronObj->Calculate();
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<codeframe::cSerializableInterface> SerializableNeuronLayer::Create(
                                                                   const std::string& className,
                                                                   const std::string& objName,
                                                                   const std::vector<codeframe::VariantValue>& params )
{
    if ( className == "SerializableNeuron" )
    {
        smart_ptr<SerializableNeuron> obj = smart_ptr<SerializableNeuron>( new SerializableNeuron( objName, NULL, 100 ) );

        (void)InsertObject( obj );

        return obj;
    }

    return smart_ptr<codeframe::cSerializableInterface>();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void SerializableNeuronLayer::SetNeuronCnt( unsigned int cnt )
{
    unsigned int thisCnt = Count();
    // Set Neuron cnt to be at least configured
    if ( cnt > thisCnt )
    {
        unsigned int newCnt = (cnt - thisCnt);
        CreateRange( "SerializableNeuron", "Neuron", newCnt );
    }
}
