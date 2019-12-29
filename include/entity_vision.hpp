#ifndef ENTITYVISION_HPP_INCLUDED
#define ENTITYVISION_HPP_INCLUDED

#include <vector>
#include <serializable_object.hpp>

#include "physics_body.hpp"
#include "colorize_circle_shape.hpp"
#include "entity_vision_node.hpp"

class EntityVision : public codeframe::Object, public EntityVisionNode
{
        CODEFRAME_META_CLASS_NAME( "EntityVision" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
         EntityVision( codeframe::ObjectNode* parent );
         EntityVision( const EntityVision& other );
        ~EntityVision();

        codeframe::Property<bool                > CastRays;
        codeframe::Property<unsigned int        > RaysCnt;
        codeframe::Property<unsigned int        > RaysSize;
        codeframe::Property<int                 > RaysStartingAngle;
        codeframe::Property<int                 > RaysEndingAngle;
        codeframe::Property< std::vector<float> > VisionVector;
        codeframe::Property< std::vector<float> > FixtureVector;

        void draw( sf::RenderTarget& target, sf::RenderStates states ) const;
        void StartFrame();
        void AddRay( EntityVision::sRay ray );
        void EndFrame();
        const std::vector<float>& GetDistanceVector();
        const std::vector<float>& GetFixtureVector();

        void SetRaysStartingAngle( int value );
        void SetRaysEndingAngle( int value );

#ifdef ENTITY_VISION_DEBUG
        void AddDirectionRay( EntityVision::sRay ray );
#endif // ENTITY_VISION_DEBUG

        void setPosition(float x, float y);
        void setRotation(float angle);
    private:
        sf::ColorizeCircleShape m_visionShape;
        std::vector<EntityVision::sRay> m_visionVector;

        void SetRaysCnt( unsigned int cnt );
        void PrepareRays();

        std::vector<float>      m_distanceVisionVector;
        std::vector<float>      m_fixtureVisionVector;
        std::vector<sf::Vertex> m_rayLines;

#ifdef ENTITY_VISION_DEBUG
        sf::Vertex         m_directionRayLine[2];
        EntityVision::sRay m_directionRay;
#endif // ENTITY_VISION_DEBUG
};

#endif // ENTITYVISION_HPP_INCLUDED
