#ifndef ENTITY_H
#define ENTITY_H

#include <serializable.hpp>

#include "entityshell.hpp"
#include "entityghost.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class Entity : public EntityGhost
{
        CODEFRAME_META_CLASS_NAME( "Entity" );
        CODEFRAME_META_BUILD_ROLE( codeframe::OBJECT );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        Entity( const std::string& name, int x, int y );
        virtual ~Entity();
        Entity(const Entity& other);
        Entity& operator=(const Entity& other);
};

#endif // ENTITY_H
