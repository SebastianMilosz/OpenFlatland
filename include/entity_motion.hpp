#ifndef ENTITY_MOTION_HPP
#define ENTITY_MOTION_HPP

#include <physics_body.hpp>

class EntityMotion : public PhysicsBody
{
    CODEFRAME_META_CLASS_NAME( "EntityMotion" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        EntityMotion(codeframe::ObjectNode* parent);
        virtual ~EntityMotion();

        codeframe::Property<float> VelocityForward;
        codeframe::Property<float> VelocityRotation;

        void synchronize( b2Body& body ) override;

    private:
        float m_velocityRotationPrew;
        float m_velocityForwardPrew;
};

#endif // ENTITY_MOTION_HPP
