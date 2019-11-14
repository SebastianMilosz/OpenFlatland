#ifndef ENTITYFACTORY_HPP
#define ENTITYFACTORY_HPP

#include <sigslot.h>
#include <serializable_object_node.hpp>
#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

#include "world.hpp"
#include "entity.hpp"

class EntityFactory : public codeframe::ObjectContainer
{
        CODEFRAME_META_CLASS_NAME( "EntityFactory" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        EntityFactory( const std::string& name, ObjectNode* parent );
        virtual ~EntityFactory();

        smart_ptr<Entity> Create( int x, int y, int z );

        signal1< smart_ptr<Entity> > signalEntityAdd;
        signal1< smart_ptr<Entity> > signalEntityDel;

        void CalculateNeuralNetworks();

    protected:
        smart_ptr<codeframe::ObjectNode> Create(
                                                             const std::string& className,
                                                             const std::string& objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                            );

    private:

};

#endif // ENTITYFACTORY_HPP
