#include "entityfactory.h"

#include <typeinfo.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::EntityFactory( std::string name, cSerializableInterface* parent ) :
    cSerializableContainer( "EntityFactory", NULL )
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
smart_ptr<Entity> EntityFactory::Create( int x, int y, int z )
{
    smart_ptr<Entity> entity = smart_ptr<Entity>( new Entity( "Unknown", x, y, z ) );

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
                                                                   const std::string className,
                                                                   const std::string objName,
                                                                   const std::vector<codeframe::VariantValue>& params )
{
    if ( className == "Entity" )
    {
        int x = 0;
        int y = 0;
        int z = 0;

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
                else if ( it->IsName( "Z" ) )
                {
                    z = it->IntegerValue();
                }
            }
        }

        smart_ptr<codeframe::cSerializableInterface> obj = smart_ptr<codeframe::cSerializableInterface>( new Entity( objName, x, y, z ) );

        InsertObject( obj );

        signalEntityAdd.Emit( obj );

        return obj;
    }

    return smart_ptr<codeframe::cSerializableInterface>();
}
