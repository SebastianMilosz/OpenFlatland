#ifndef ENTITY_VISION_NODE_H
#define ENTITY_VISION_NODE_H

#include <Box2D/Box2D.h>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>

class EntityVisionNode
{
    public:
        struct Ray
        {
            Ray();
            Ray( const b2Vec2& p1, const b2Vec2& p2, const uint32_t f );

            b2Vec2 P1;
            b2Vec2 P2;
            uint32_t Fixture;
        };

        struct RayData
        {
            RayData();
            RayData(const float distance, const uint32_t fixture);

            bool operator==(const RayData& rval) const
            {
                if (  std::fabs(Distance - rval.Distance) > 0.001F && Fixture == rval.Fixture )
                {
                    return true;
                }
                return false;
            }

            float    Distance;
            uint32_t Fixture;
        };

                 EntityVisionNode();
        virtual ~EntityVisionNode() = default;

        virtual const std::vector<RayData>& GetVisionVector() const = 0;
};

#endif // ENTITY_VISION_NODE_H
