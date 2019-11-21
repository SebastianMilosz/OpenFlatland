#ifndef WORLD_H
#define WORLD_H

#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>
#include <serializable_object.hpp>

#include "entity.hpp"
#include "constelement.hpp"

class World : public codeframe::Object
{
        CODEFRAME_META_CLASS_NAME( "World" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 World( const std::string& name, ObjectNode* parent );
        virtual ~World();

        void AddShell( std::shared_ptr<Entity>       entity );
        void AddConst( std::shared_ptr<ConstElement> constElement );

        bool PhysisStep(sf::RenderWindow& window);
        bool Draw( sf::RenderWindow& window );

        void MouseDown( const float x, const float y );
        void MouseUp( const float x, const float y );
        void MouseMove( const float x, const float y );

    protected:

    private:
        class RayCastCallback : public b2RayCastCallback
        {
            public:
                RayCastCallback()
                {
                    m_hit = false;
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

        b2Body* getBodyAtMouse( const float x, const float y );

        void CalculateRays( void );

        b2Body*         m_GroundBody;
        b2MouseJoint*   m_MouseJoint;
        b2MouseJointDef m_JointDef;
        b2Vec2          m_Gravity;
        b2World         m_World;

        bool            m_entitySelMode;
};

#endif // WORLD_H
