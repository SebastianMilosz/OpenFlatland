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
         EntityVision(codeframe::ObjectNode* parent);
         EntityVision(const EntityVision& other);
        ~EntityVision();

        codeframe::Property<bool                > CastRays;
        codeframe::Property<unsigned int        > RaysCnt;
        codeframe::Property<unsigned int        > RaysSize;
        codeframe::Property<int                 > RaysStartingAngle;
        codeframe::Property<int                 > RaysEndingAngle;
        codeframe::Property< std::vector<float> > VisionVector;
        codeframe::Property< std::vector<float> > FixtureVector;

        const std::vector<float>& GetDistanceVector();
        const std::vector<float>& GetFixtureVector();

        virtual void StartFrame();
        virtual void AddRay(EntityVision::sRay ray);
        virtual void EndFrame();

        virtual void SetRaysStartingAngle(const int value);
        virtual void SetRaysEndingAngle(const int value);

        virtual void setPosition(const float x, const float y);
        virtual void setRotation(const float angle);

#ifdef ENTITY_VISION_DEBUG
        virtual void AddDirectionRay(EntityVision::sRay ray);
#endif // ENTITY_VISION_DEBUG

    protected:
        virtual void SetRaysCnt(const unsigned int cnt);

        std::vector<EntityVision::sRay> m_visionVector;
        std::vector<float> m_distanceVisionVector;
        std::vector<float> m_fixtureVisionVector;
};

#endif // ENTITYVISION_HPP_INCLUDED
