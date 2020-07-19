#ifndef DRAWABLE_ENTITY_ENERGY_HPP_INCLUDED
#define DRAWABLE_ENTITY_ENERGY_HPP_INCLUDED

#include "entity_energy.hpp"
#include "drawable_object.hpp"

class DrawableEntityEnergy : public EntityEnergy, public DrawableObject
{
    public:
        DrawableEntityEnergy(codeframe::ObjectNode* parent);
        virtual ~DrawableEntityEnergy() = default;

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;
    protected:

    private:
};

#endif // DRAWABLE_ENTITY_ENERGY_HPP_INCLUDED
