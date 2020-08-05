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
    VelocityForward ( this, "VelocityForward" , 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("VelocityForward") ),
    VelocityRotation( this, "VelocityRotation", 0.0F , cPropertyInfo().Kind(KIND_REAL).Description("VelocityRotation") ),
    MotionVector    ( this, "MotionVector"    , thrust::host_vector<float>(), cPropertyInfo().Kind(KIND_VECTOR_THRUST_HOST, KIND_REAL).Description("MotionVector")),
    EnergyConsumer  ( this, "EnergyConsumer"  , 0.0F , cPropertyInfo().Kind(KIND_REAL).ReferencePath("../Energy.Energy").Description("EnergyConsumer") )
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
        if ((float)EnergyConsumer > 0.0)
        {
            float  angle(-body.GetAngle());
            b2Vec2 velocityDirection(std::sin(angle) * velocityForward , std::cos(angle) * velocityForward);

            body.SetLinearVelocity(velocityDirection);
            body.SetAngularVelocity(velocityRotation * DEGTORAD);

            m_velocityRotationPrew = velocityRotation;
            m_velocityForwardPrew = velocityForward;

            EnergyConsumer = (float)EnergyConsumer - 1.0;
        }
    }
    else if (std::fabs(m_velocityRotationPrew) > 0.0F || std::fabs(m_velocityForwardPrew) > 0.0F)
    {
        body.SetLinearVelocity(b2Vec2(0.0F,0.0F));
        body.SetAngularVelocity(0.0F);
        m_velocityRotationPrew = 0.0F;
        m_velocityForwardPrew = 0.0F;
    }
}
