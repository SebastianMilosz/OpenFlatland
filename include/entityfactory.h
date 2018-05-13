#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include "world.h"
#include "entity.h"

#include <serializable.h>
#include <serializablecontainer.h>

class EntityFactory : public codeframe::cSerializableContainer<Entity>
{
    public:
        std::string Role()      const { return "Object";     }
        std::string Class()     const { return "MainWindow"; }
        std::string BuildType() const { return "Static";     }

    public:
        EntityFactory( World& world );
        virtual ~EntityFactory();

        void Save( std::string file );
        void Load( std::string file );

        smart_ptr<Entity> Create( int x, int y, int z );

    protected:
        smart_ptr<Entity> Create( std::string className, std::string objName, int cnt );

    private:
        World&                               m_world;
};

#endif // ENTITYFACTORY_H
