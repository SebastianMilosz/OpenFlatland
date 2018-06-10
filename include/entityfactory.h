#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include "world.h"
#include "entity.h"

#include <serializable.h>
#include <serializablecontainer.h>
#include <serializableinterface.h>

class EntityFactory : public codeframe::cSerializableContainer<codeframe::cSerializableInterface>
{
    public:
        std::string Role()      const { return "Container";     }
        std::string Class()     const { return "EntityFactory"; }
        std::string BuildType() const { return "Static";        }

    public:
        EntityFactory( World& world );
        virtual ~EntityFactory();

        smart_ptr<Entity> Create( int x, int y, int z );

    protected:
        smart_ptr<codeframe::cSerializableInterface> Create( std::string className, std::string objName, int cnt );

    private:
        World&                               m_world;
};

#endif // ENTITYFACTORY_H
