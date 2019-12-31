#ifndef ENTITYVISION_HPP_INCLUDED
#define ENTITYVISION_HPP_INCLUDED

#include <vector>
#include <Box2D/Box2D.h>
#include <serializable_object.hpp>

#include "physics_body.hpp"
#include "colorize_circle_shape.hpp"
#include "entity_vision_node.hpp"

class EntityVision : public codeframe::Object, public EntityVisionNode, public sf::Transformable
{
        CODEFRAME_META_CLASS_NAME( "EntityVision" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
         EntityVision(codeframe::ObjectNode* parent);
         EntityVision(const EntityVision& other);
        ~EntityVision();

        codeframe::Property<bool                > DrawRays;
        codeframe::Property<unsigned int        > RaysCnt;
        codeframe::Property<unsigned int        > RaysSize;
        codeframe::Property<int                 > RaysStartingAngle;
        codeframe::Property<int                 > RaysEndingAngle;
        codeframe::Property< std::vector<float> > VisionVector;
        codeframe::Property< std::vector<float> > FixtureVector;

        const std::vector<float>& GetDistanceVector();
        const std::vector<float>& GetFixtureVector();

        void CastRays(b2World& world, const b2Vec2& p1);

        virtual void StartFrame();
        virtual void AddRay(EntityVision::sRay ray);
        virtual void EndFrame();

        virtual void SetRaysStartingAngle(const int value);
        virtual void SetRaysEndingAngle(const int value);

        virtual void setPosition(float x, float y);
        virtual void setRotation(float angle);

#ifdef ENTITY_VISION_DEBUG
        virtual void AddDirectionRay(EntityVision::sRay ray);
#endif // ENTITY_VISION_DEBUG

    protected:
        virtual void SetRaysCnt(const unsigned int cnt);

        std::vector<EntityVision::sRay> m_visionVector;
        std::vector<float> m_distanceVisionVector;
        std::vector<float> m_fixtureVisionVector;

    private:
        class RayCastCallback : public b2RayCastCallback
        {
            public:
                inline void Reset()
                {
                    m_hit = false;
                }

                RayCastCallback()
                {
                    Reset();
                }

                float32 ReportFixture( b2Fixture* fixture, const b2Vec2& point,
                                       const b2Vec2& normal, const float32 fraction )
                {
                    m_hit    = true;
                    m_point  = point;
                    m_normal = normal;

                    return fraction;
                }

                bool   WasHit() const { return m_hit; }
                b2Vec2 HitPoint() const { return m_point; }

            private:
                bool m_hit;
                b2Vec2 m_point;
                b2Vec2 m_normal;
        };

        RayCastCallback m_rayCastCallback;
};

#endif // ENTITYVISION_HPP_INCLUDED
