#include "entityfactory.h"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::EntityFactory( b2World& world ) :
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
std::shared_ptr<Entity> EntityFactory::Create( int x, int y, int z )
{
    std::shared_ptr<Entity> entity = std::make_shared<Entity>();

    // Factory production line
    entity->AddShell( std::make_shared<EntityShell>( m_world, x, y, z ) );

    m_entityList.push_back( entity );

    return entity;
}


