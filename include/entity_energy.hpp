#ifndef ENTITY_ENERGY_HPP_INCLUDED
#define ENTITY_ENERGY_HPP_INCLUDED

#include <physics_body.hpp>

class EntityEnergy : public PhysicsBody
{
    CODEFRAME_META_CLASS_NAME( "EntityEnergy" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        EntityEnergy(codeframe::ObjectNode* parent);
        virtual ~EntityEnergy() = default;

        codeframe::Property<float> Energy;

        void synchronize( b2Body& body ) override;
};

#endif // ENTITY_ENERGY_HPP_INCLUDED
