#include "entity_vision_node.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVisionNode::Ray::Ray() :
    P1( 0.0F, 0.0F ),
    P2( 0.0F, 0.0F ),
    Fixture( 0.0F )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVisionNode::Ray::Ray( const b2Vec2& p1, const b2Vec2& p2, const uint32_t f ) :
    P1( p1 ),
    P2( p2 ),
    Fixture( f )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVisionNode::EntityVisionNode()
{
    //ctor
}
