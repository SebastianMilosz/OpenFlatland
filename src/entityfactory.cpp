#include "entityfactory.h"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::EntityFactory( World& world ) :
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
    std::shared_ptr<Entity> entity = std::make_shared<Entity>( x, y, z );

    m_entityList.push_back( entity );

    return entity;
}


