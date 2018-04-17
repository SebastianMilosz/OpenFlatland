#ifndef ENTITYGHOST_H
#define ENTITYGHOST_H

#include "entityshell.h"

class EntityGhost : public EntityShell
{
    public:
        EntityGhost( int x, int y, int z );
        virtual ~EntityGhost();
        EntityGhost(const EntityGhost& other);
        EntityGhost& operator=(const EntityGhost& other);

    protected:

    private:
};

#endif // ENTITYGHOST_H
