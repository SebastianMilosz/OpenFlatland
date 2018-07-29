#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include "world.h"
#include "entity.h"

#include <serializable.h>
#include <serializablecontainer.h>
#include <serializableinterface.h>

class EntityFactory : public codeframe::cSerializableContainer
{
    public:
        std::string Role()            const { return "Container";     }
        std::string Class()           const { return "EntityFactory"; }
        std::string BuildType()       const { return "Static";        }
        std::string ConstructPatern() const { return ""; }

    public:
        EntityFactory( World& world );
        virtual ~EntityFactory();

        smart_ptr<Entity> Create( int x, int y, int z );

    protected:
        smart_ptr<codeframe::cSerializableInterface> Create(
                                                             const std::string className,
                                                             const std::string objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                            );

    private:
        World&                               m_world;
};

#endif // ENTITYFACTORY_H
