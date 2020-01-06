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

        codeframe::Property<bool                  > DrawRays;
        codeframe::Property<unsigned int          > RaysCnt;
        codeframe::Property<unsigned int          > RaysSize;
        codeframe::Property<int                   > RaysStartingAngle;
        codeframe::Property<int                   > RaysEndingAngle;
        codeframe::Property< std::vector<RayData> > VisionVector;

        const std::vector<RayData>& GetVisionVector() const override;

        void CastRays(b2World& world, const b2Vec2& p1);

        virtual void setPosition(float x, float y);
        virtual void setRotation(float angle);

    protected:
        virtual void SetRaysStartingAngle(const int value);
        virtual void SetRaysEndingAngle(const int value);

        virtual void StartFrame();
        virtual void AddRay(EntityVision::Ray ray);
        virtual void EndFrame();

#ifdef ENTITY_VISION_DEBUG
        virtual void AddDirectionRay(EntityVision::Ray ray);
#endif // ENTITY_VISION_DEBUG

        virtual void SetRaysCnt(const unsigned int cnt);

        std::vector<EntityVision::Ray> m_visionVector;
        std::vector<EntityVision::RayData> m_visionDataVector;

    private:
        static constexpr float NUMBER_PI = 3.141592654F;
        static constexpr float TO_RADIAN(const float dg) { return dg * (NUMBER_PI/180.0F); }

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

                bool     WasHit() const { return m_hit; }
                b2Vec2   HitPoint() const { return m_point; }
                uint32_t Fixture() const { return 0U; }

            private:
                bool m_hit;
                b2Vec2 m_point;
                b2Vec2 m_normal;
        };

        RayCastCallback m_rayCastCallback;
};

#endif // ENTITYVISION_HPP_INCLUDED
