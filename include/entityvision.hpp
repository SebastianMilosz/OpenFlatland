#ifndef ENTITYVISION_HPP_INCLUDED
#define ENTITYVISION_HPP_INCLUDED

#include <vector>
#include <serializable_object.hpp>

#include "physicsbody.hpp"

class EntityVision : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME( "EntityVision" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        EntityVision( codeframe::ObjectNode* parent );
        ~EntityVision();

        codeframe::Property< std::vector<float>, EntityVision > VisionVector;
        codeframe::Property< std::vector<float>, EntityVision > FixtureVector;

        struct sRay
        {
            sRay();
            sRay( b2Vec2& p1, b2Vec2& p2, float32 f );

            b2Vec2 P1;
            b2Vec2 P2;
            float32 Fixture;
        };

        void Draw( sf::RenderWindow& window );
        void StartFrame();
        void AddRay( EntityVision::sRay ray );
        void AddDirectionRay( EntityVision::sRay ray );
        void EndFrame();

        const std::vector<float>& GetDistanceVector();
        const std::vector<float>& GetFixtureVector();

    private:
        std::vector<EntityVision::sRay> m_visionVector;

        std::vector<float> m_distanceVisionVector;
        std::vector<float> m_fixtureVisionVector;
        sf::Vertex         m_rayLine[2];
        sf::Vertex         m_directionRayLine[2];
        EntityVision::sRay m_directionRay;
};

#endif // ENTITYVISION_HPP_INCLUDED
