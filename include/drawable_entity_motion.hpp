#ifndef DRAWABLE_ENTITY_MOTION_HPP
#define DRAWABLE_ENTITY_MOTION_HPP

#include "entity_motion.hpp"
#include "drawable_object.hpp"

class DrawableEntityMotion : public EntityMotion, public DrawableObject
{
    public:
        DrawableEntityMotion(codeframe::ObjectNode* parent);
        virtual ~DrawableEntityMotion();

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;
    protected:

    private:
};

#endif // DRAWABLE_ENTITY_MOTION_HPP
