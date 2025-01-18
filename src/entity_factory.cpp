#include "entity_factory.hpp"

#include <typeinfo.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::EntityFactory( const std::string& name, ObjectNode* parent ) :
    ObjectContainer( name, parent )
{
    //ctor
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
        smart_ptr<ObjectNode> serializableObj = Get( n );

        Entity* entityObj = static_cast<Entity*>( smart_ptr_getRaw( serializableObj ) );

        if ( (Entity*)nullptr != entityObj )
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
smart_ptr<codeframe::Object> EntityFactory::Create(
                                                        const std::string& className,
                                                        const std::string& objName,
                                                        const std::vector<codeframe::VariantValue>& params )
{
    if ( className == "Entity" )
    {
        int x( 0 );
        int y( 0 );

        for ( auto it = params.begin(); it != params.end(); ++it )
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

    return smart_ptr<codeframe::Object>();
}
