#ifndef ENTITYVISION_HPP_INCLUDED
#define ENTITYVISION_HPP_INCLUDED

#include <vector>
#include <memory>
#include <serializable_object.hpp>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>

#include "physicsbody.hpp"
#include "colorizecircleshape.hpp"

class EntityVision : public codeframe::Object, public sf::Drawable, public sf::Transformable
{
        CODEFRAME_META_CLASS_NAME( "EntityVision" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
         EntityVision( codeframe::ObjectNode* parent );
         EntityVision( const EntityVision& other );
        ~EntityVision();

        codeframe::Property<bool,                EntityVision > CastRays;
        codeframe::Property<unsigned int,        EntityVision > RaysCnt;
        codeframe::Property<unsigned int,        EntityVision > RaysSize;
        codeframe::Property<int,                 EntityVision > RaysStartingAngle;
        codeframe::Property<int,                 EntityVision > RaysEndingAngle;
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

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const;
        void StartFrame();
        void AddRay( EntityVision::sRay ray );
        void EndFrame();
        const std::vector<float>& GetDistanceVector();
        const std::vector<float>& GetFixtureVector();

        void SetRaysStartingAngle( int value );
        void SetRaysEndingAngle( int value );

        void setPosition(float x, float y);
        void setRotation(float angle);

#ifdef ENTITY_VISION_DEBUG
        void AddDirectionRay( EntityVision::sRay ray );
#endif // ENTITY_VISION_DEBUG

    private:
        sf::ColorizeCircleShape m_visionShape;
        std::vector<EntityVision::sRay> m_visionVector;

        void SetRaysCnt( unsigned int cnt );
        void PrepareRays();

        std::vector<float>          m_distanceVisionVector;
        std::vector<float>          m_fixtureVisionVector;
        std::unique_ptr<sf::Vertex> m_rayLines;

        float m_x;
        float m_y;
        float m_r;

#ifdef ENTITY_VISION_DEBUG
        sf::Vertex         m_directionRayLine[2];
        EntityVision::sRay m_directionRay;
#endif // ENTITY_VISION_DEBUG
};

#endif // ENTITYVISION_HPP_INCLUDED
