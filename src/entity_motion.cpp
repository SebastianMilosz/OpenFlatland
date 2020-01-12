#include "entity_motion.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityMotion::EntityMotion(codeframe::ObjectNode* parent) :
    PhysicsBody("Motion", parent),
    VelocityX( this, "VelocityX", 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("VelocityX") ),
    VelocityY( this, "VelocityY", 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("VelocityY") )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityMotion::~EntityMotion()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityMotion::synchronize( b2Body& body )
{
    b2Vec2 vel = body.GetLinearVelocity();

    vel.x = (float)VelocityX;
    vel.y = (float)VelocityY;

    body.SetLinearVelocity( vel );
}
