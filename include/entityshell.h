#ifndef ENTITYSHELL_H
#define ENTITYSHELL_H

#include "world.h"

#include <Box2D/Box2D.h>

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

    protected:

    private:
        b2Body* m_Body;
};

#endif // ENTITYSHELL_H
