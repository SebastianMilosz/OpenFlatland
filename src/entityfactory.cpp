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
void EntityFactory::Save( std::string file )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityFactory::Load( std::string file )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<Entity> EntityFactory::Create( int x, int y, int z )
{
    smart_ptr<Entity> entity = smart_ptr<Entity>( new Entity( "Unknown", x, y, z ) );

    //m_entityList.push_back( entity );
    InsertObject( entity, 1 );

    return entity;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<Entity> EntityFactory::Create( std::string className, std::string objName, int cnt )
{
    if( className == "Entity" )
    {
        smart_ptr<Entity> obj = smart_ptr<Entity>( new Entity( objName, 0, 0, 0 ) );

        InsertObject( obj, cnt );
        return obj;
    }

    return smart_ptr<Entity>();
}
