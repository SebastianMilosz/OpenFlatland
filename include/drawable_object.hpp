#ifndef DRAWABLE_OBJECT_H
#define DRAWABLE_OBJECT_H

#include <SFML/Graphics.hpp>

class DrawableObject : public sf::Drawable
{
    public:
        DrawableObject();
        virtual ~DrawableObject() = default;

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const override;
    protected:
        static void appendQuad(sf::VertexArray& vec, const sf::Vector2f position, const sf::Vector2f size, const sf::Color color);

    private:
};

#endif // DRAWABLE_OBJECT_H
