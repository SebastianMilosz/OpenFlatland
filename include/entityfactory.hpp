#ifndef ENTITYFACTORY_HPP
#define ENTITYFACTORY_HPP

#include <sigslot.h>
#include <serializable.hpp>
#include <serializablecontainer.hpp>
#include <serializableinterface.hpp>

#include "world.hpp"
#include "entity.hpp"

class EntityFactory : public codeframe::cSerializableContainer
{
        CODEFRAME_META_CLASS_NAME( "EntityFactory" );
        CODEFRAME_META_BUILD_ROLE( codeframe::CONTAINER );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        EntityFactory( std::string name, cSerializableInterface* parent );
        virtual ~EntityFactory();

        smart_ptr<Entity> Create( int x, int y, int z );

        signal1< smart_ptr<Entity> > signalEntityAdd;
        signal1< smart_ptr<Entity> > signalEntityDel;

        void CalculateNeuralNetworks();

    protected:
        smart_ptr<codeframe::cSerializableInterface> Create(
                                                             const std::string& className,
                                                             const std::string& objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                            );

    private:

};

#endif // ENTITYFACTORY_HPP
