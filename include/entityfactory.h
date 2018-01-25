#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include "world.h"
#include "entity.h"

#include <list>

class EntityFactory
{
    public:
        EntityFactory( World& world );
        virtual ~EntityFactory();

        std::shared_ptr<Entity> Create( int x, int y, int z );

    protected:

    private:
        World&                               m_world;
        std::list< std::shared_ptr<Entity> > m_entityList;
};

#endif // ENTITYFACTORY_H
