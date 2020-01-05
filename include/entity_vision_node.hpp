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
            Ray( const b2Vec2& p1, const b2Vec2& p2, const float32 f );

            b2Vec2 P1;
            b2Vec2 P2;
            float32 Fixture;
        };

        struct RayData
        {
            RayData();
            RayData(float distance, uint32_t fixture);

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
        virtual ~EntityVisionNode();

        virtual const std::vector<RayData>& GetVisionVector() const = 0;
};

#endif // ENTITY_VISION_NODE_H
