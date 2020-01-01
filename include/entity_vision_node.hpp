#ifndef ENTITY_VISION_NODE_H
#define ENTITY_VISION_NODE_H

#include <Box2D/Box2D.h>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>

class EntityVisionNode
{
    public:
        struct sRay
        {
            sRay();
            sRay( const b2Vec2& p1, const b2Vec2& p2, const float32 f );

            b2Vec2 P1;
            b2Vec2 P2;
            float32 Fixture;
        };

                 EntityVisionNode();
        virtual ~EntityVisionNode();

        virtual const std::vector<float>& GetDistanceVector() const = 0;
        virtual const std::vector<float>& GetFixtureVector() const = 0;
    protected:

    private:
};

#endif // ENTITY_VISION_NODE_H
