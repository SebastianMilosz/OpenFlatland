#ifndef ENTITY_VISION_NODE_H
#define ENTITY_VISION_NODE_H

#include "ray_data.hpp"

#include <box2d/box2d.h>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>
#include <thrust/device_vector.h>

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

                 EntityVisionNode();
        virtual ~EntityVisionNode() = default;

        virtual const thrust::host_vector<RayData>& GetConstVisionVector() const = 0;
        virtual thrust::host_vector<RayData>& GetVisionVector() = 0;
};

#endif // ENTITY_VISION_NODE_H
