#ifndef ENTITY_MOTION_HPP
#define ENTITY_MOTION_HPP

#include <physics_body.hpp>
#include <thrust/device_vector.h>

class EntityMotion : public PhysicsBody
{
    CODEFRAME_META_CLASS_NAME( "EntityMotion" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        EntityMotion(codeframe::ObjectNode* parent);
        virtual ~EntityMotion() = default;

        codeframe::Property<float> VelocityForward;
        codeframe::Property<float> VelocityRotation;
        codeframe::Property< thrust::host_vector<float> > MotionVector;
        codeframe::Property<float> EnergyConsumer;

        void synchronize( b2Body& body ) override;

    private:
        float m_velocityRotationPrew;
        float m_velocityForwardPrew;
};

#endif // ENTITY_MOTION_HPP
