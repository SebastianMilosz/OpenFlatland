#ifndef DRAWABLE_ENTITY_MOTION_HPP
#define DRAWABLE_ENTITY_MOTION_HPP

#include <entity_motion.hpp>

class DrawableEntityMotion : public EntityMotion, public sf::Drawable
{
    public:
        DrawableEntityMotion(codeframe::ObjectNode* parent);
        virtual ~DrawableEntityMotion();

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;
    protected:

    private:
};

#endif // DRAWABLE_ENTITY_MOTION_HPP
