#ifndef WORLD_H
#define WORLD_H

#include <SFML/Graphics.hpp>
#include <Box2D/Box2D.h>

class World
{
    public:
        World();
        virtual ~World();

        b2Body* CreateBody( b2BodyDef* def );

        bool PhysisStep();
        bool Draw( sf::RenderWindow& window );
        bool MouseDown( float x, float y );

    protected:

    private:
        b2Body* getBodyAtMouse( float x, float y );

        b2Vec2   m_Gravity;
        b2World  m_World;
        sf::Font m_font;
};

#endif // WORLD_H
