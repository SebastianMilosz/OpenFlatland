#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include "world.h"
#include "entity.h"

#include <sigslot.h>
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
        EntityFactory( std::string name, cSerializableInterface* parent );
        virtual ~EntityFactory();

        smart_ptr<Entity> Create( int x, int y, int z );

        signal1< smart_ptr<Entity> > signalEntityAdd;
        signal1< smart_ptr<Entity> > signalEntityDel;

    protected:
        smart_ptr<codeframe::cSerializableInterface> Create(
                                                             const std::string className,
                                                             const std::string objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                            );

    private:

};

#endif // ENTITYFACTORY_H
