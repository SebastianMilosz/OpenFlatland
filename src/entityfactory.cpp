#include "entityfactory.h"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::EntityFactory( World& world ) :
    cSerializableContainer( "EntityFactory", NULL ),
    m_world( world )
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

    return entity;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<codeframe::cSerializableInterface> EntityFactory::Create( std::string className, std::string objName, int cnt )
{
    if( className == "Entity" )
    {
        smart_ptr<codeframe::cSerializableInterface> obj = smart_ptr<codeframe::cSerializableInterface>( new Entity( objName, 0, 0, 0 ) );

        InsertObject( obj, cnt );
        return obj;
    }

    return smart_ptr<codeframe::cSerializableInterface>();
}
