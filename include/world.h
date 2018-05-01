#ifndef WORLD_H
#define WORLD_H

#include <serializable.h>
#include <SFML/Graphics.hpp>
#include <Box2D/Box2D.h>

#include "entityshell.h"

class World
{
    public:
                 World();
        virtual ~World();

        bool AddShell( std::shared_ptr<EntityShell> shell );

        bool PhysisStep();
        bool Draw( sf::RenderWindow& window );

        void MouseDown( float x, float y );
        void MouseUp( float x, float y );
        void MouseMove( float x, float y );

    protected:

    private:
        b2Body* getBodyAtMouse( float x, float y );

        b2Body*         m_GroundBody;
        b2MouseJoint*   m_MouseJoint;
        b2MouseJointDef m_JointDef;
        b2Vec2          m_Gravity;
        b2World         m_World;
        sf::Font        m_font;

        bool            m_entitySelMode;
};

#endif // WORLD_H
