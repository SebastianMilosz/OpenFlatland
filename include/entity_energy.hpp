#ifndef ENTITY_ENERGY_HPP_INCLUDED
#define ENTITY_ENERGY_HPP_INCLUDED

#include <physics_body.hpp>
#include <thrust/device_vector.h>

class EntityEnergy : public PhysicsBody
{
    CODEFRAME_META_CLASS_NAME( "EntityEnergy" );
    CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        EntityEnergy(codeframe::ObjectNode* parent);
        virtual ~EntityEnergy() = default;

        codeframe::Property<float> Energy;
        codeframe::Property< thrust::host_vector<float> > EnergyVector;

        const thrust::host_vector<float>& GetConstEnergyVector() const;
              thrust::host_vector<float>& GetEnergyVector();

        void synchronize( b2Body& body ) override;

    private:
        thrust::host_vector<float> m_energyDataVector;
};

#endif // ENTITY_ENERGY_HPP_INCLUDED
