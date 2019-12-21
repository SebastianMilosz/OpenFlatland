#ifndef ENTITY_VISION_NODE_H
#define ENTITY_VISION_NODE_H

#include <Box2D/Box2D.h>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>

class EntityVisionNode : public sf::Drawable, public sf::Transformable
{
    public:
        struct sRay
        {
            sRay();
            sRay( b2Vec2& p1, b2Vec2& p2, float32 f );

            b2Vec2 P1;
            b2Vec2 P2;
            float32 Fixture;
        };

                 EntityVisionNode();
        virtual ~EntityVisionNode();

    protected:

    private:
};

#endif // ENTITY_VISION_NODE_H
