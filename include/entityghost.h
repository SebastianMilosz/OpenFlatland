#ifndef ENTITYGHOST_H
#define ENTITYGHOST_H

#include "world.h"

#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>

class EntityGhost
{
    public:
        EntityGhost( World& World, int x, int y, int z );
        virtual ~EntityGhost();
        EntityGhost(const EntityGhost& other);
        EntityGhost& operator=(const EntityGhost& other);

    protected:

    private:
};

#endif // ENTITYGHOST_H
