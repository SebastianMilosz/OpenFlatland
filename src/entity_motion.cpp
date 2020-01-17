#include "entity_motion.hpp"

#include <cmath>

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityMotion::EntityMotion(codeframe::ObjectNode* parent) :
    PhysicsBody("Motion", parent),
    VelocityForward( this, "VelocityForward"  , 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("VelocityForward") ),
    VelocityRotation( this, "VelocityRotation", 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("VelocityRotation") )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityMotion::synchronize(b2Body& body)
{
    static const float DEGTORAD = 0.0174532925199432957F;

    const float velocityRotation((float)VelocityRotation);
    const float velocityForward((float)VelocityForward);

    if (std::fabs(velocityRotation) > 0.0F || std::fabs(velocityForward) > 0.0F)
    {
        float  angle(-body.GetAngle());
        b2Vec2 velocityDirection(std::sin(angle) * velocityForward , std::cos(angle) * velocityForward);

        body.SetLinearVelocity(velocityDirection);
        body.SetAngularVelocity(velocityRotation * DEGTORAD);

        m_velocityRotationPrew = velocityRotation;
        m_velocityForwardPrew = velocityForward;
    }
    else if (std::fabs(m_velocityRotationPrew) > 0.0F || std::fabs(m_velocityForwardPrew) > 0.0F)
    {
        body.SetLinearVelocity(b2Vec2(0.0F,0.0F));
        body.SetAngularVelocity(0.0F);
        m_velocityRotationPrew = 0.0F;
        m_velocityForwardPrew = 0.0F;
    }
}
