#include "entityfactory.hpp"

#include <typeinfo.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::EntityFactory( std::string name, cSerializableInterface* parent ) :
    cSerializableContainer( name, parent )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::~EntityFactory()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityFactory::CalculateNeuralNetworks()
{
    for ( unsigned int n = 0; n < Count(); n++ )
    {
        smart_ptr<cSerializableInterface> serializableObj = Get( n );

        Entity* entityObj = static_cast<Entity*>( smart_ptr_getRaw( serializableObj ) );

        if ( (Entity*)NULL != entityObj )
        {
            entityObj->CalculateNeuralNetworks();
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<Entity> EntityFactory::Create( int x, int y, int z )
{
    smart_ptr<Entity> entity = smart_ptr<Entity>( new Entity( "Unknown", x, y ) );

    InsertObject( entity );

    signalEntityAdd.Emit( entity );

    return entity;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<codeframe::cSerializableInterface> EntityFactory::Create(
                                                                   const std::string& className,
                                                                   const std::string& objName,
                                                                   const std::vector<codeframe::VariantValue>& params )
{
    if ( className == "Entity" )
    {
        int x = 0;
        int y = 0;

        for ( std::vector<codeframe::VariantValue>::const_iterator it = params.begin(); it != params.end(); ++it )
        {
            if ( it->GetType() == codeframe::TYPE_INT )
            {
                     if ( it->IsName( "X" ) )
                {
                    x = it->IntegerValue();
                }
                else if ( it->IsName( "Y" ) )
                {
                    y = it->IntegerValue();
                }
            }
        }

        smart_ptr<Entity> obj = smart_ptr<Entity>( new Entity( objName, x, y ) );

        (void)InsertObject( obj );

        signalEntityAdd.Emit( obj );

        return obj;
    }

    return smart_ptr<codeframe::cSerializableInterface>();
}
