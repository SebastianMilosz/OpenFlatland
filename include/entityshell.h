#ifndef ENTITYSHELL_H
#define ENTITYSHELL_H

#include "world.h"

#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>

class EntityShell
{
    public:
        EntityShell( World& World, int x, int y, int z );
        virtual ~EntityShell();
        EntityShell(const EntityShell& other);
        EntityShell& operator=(const EntityShell& other);

        unsigned int GetX();
        void SetX(unsigned int val);
        unsigned int GetY();
        void SetY(unsigned int val);
        unsigned int GetZ();
        void SetZ(unsigned int val);

        void SetColor( const sf::Color& color ) { m_color = color; }
        sf::Color& GetColor() { return m_color; }

    protected:

    private:
        b2Body*     m_Body;
        sf::Color   m_color;
};

#endif // ENTITYSHELL_H
