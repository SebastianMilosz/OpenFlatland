#ifndef ENTITYVISION_HPP_INCLUDED
#define ENTITYVISION_HPP_INCLUDED

#include <vector>

#include "physicsbody.hpp"

class EntityVision
{
    public:
        EntityVision();
        ~EntityVision();

        struct sRay
        {
            sRay( b2Vec2& p1, b2Vec2& p2, float32 f );

            b2Vec2 P1;
            b2Vec2 P2;
            float32 Fixture;
        };

        void Draw( sf::RenderWindow& window );
        void StartFrame();
        void AddRay( EntityVision::sRay ray );
        void EndFrame();

    private:
        std::vector<EntityVision::sRay> m_visionVector;
        sf::Vertex                      m_rayLine[2];
};

#endif // ENTITYVISION_HPP_INCLUDED
