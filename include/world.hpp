#ifndef WORLD_H
#define WORLD_H

#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>
#include <serializable.hpp>

#include "entity.hpp"
#include "constelement.hpp"

class World : public codeframe::cSerializable
{
    public:
        std::string Role()            const { return "Object"; }
        std::string Class()           const { return "World";  }
        std::string BuildType()       const { return "Static"; }
        std::string ConstructPatern() const { return "";       }

    public:
                 World( const std::string& name, cSerializableInterface* parent );
        virtual ~World();

        void AddShell( std::shared_ptr<Entity>       entity );
        void AddConst( std::shared_ptr<ConstElement> constElement );

        bool PhysisStep(sf::RenderWindow& window);
        bool Draw( sf::RenderWindow& window );

        void MouseDown( float x, float y );
        void MouseUp( float x, float y );
        void MouseMove( float x, float y );

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
                                       const b2Vec2& normal, float32 fraction )
                {
                    m_hit    = true;
                    m_point  = point;
                    m_normal = normal;

                    return fraction;
                }

                bool   WasHit() { return m_hit; }
                b2Vec2 HitPoint() { return m_point; }

            private:
                bool m_hit;
                b2Vec2 m_point;
                b2Vec2 m_normal;
        };

        b2Body* getBodyAtMouse( float x, float y );

        void CalculateRays( void );

        b2Body*         m_GroundBody;
        b2MouseJoint*   m_MouseJoint;
        b2MouseJointDef m_JointDef;
        b2Vec2          m_Gravity;
        b2World         m_World;

        bool            m_entitySelMode;
};

#endif // WORLD_H
