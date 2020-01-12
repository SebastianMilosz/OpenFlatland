#ifndef DRAWABLE_OBJECT_H
#define DRAWABLE_OBJECT_H

#include <SFML/Graphics.hpp>

class DrawableObject : public sf::Drawable
{
    public:
        DrawableObject();
        virtual ~DrawableObject();

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;
    protected:

    private:
};

#endif // DRAWABLE_OBJECT_H
