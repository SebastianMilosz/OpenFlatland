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
    VelocityY( this, "VelocityY", 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("VelocityY") ),
    VelocityR( this, "VelocityR", 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("VelocityR") )
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
    #define DEGTORAD 0.0174532925199432957f

    b2Vec2 vel = body.GetLinearVelocity();
    float omega = (float)VelocityR * DEGTORAD;

    vel.x = (float)VelocityX;
    vel.y = (float)VelocityY;

    body.SetLinearVelocity(vel);
    body.SetAngularVelocity(omega);
}
