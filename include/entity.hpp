#ifndef ENTITY_H
#define ENTITY_H

#include <serializable_object.hpp>

#include "entity_shell.hpp"
#include "entity_ghost.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class Entity : public EntityGhost
{
        CODEFRAME_META_CLASS_NAME( "Entity" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        Entity( const std::string& name, int x, int y );
        virtual ~Entity();
        Entity(const Entity& other);
        Entity& operator=(const Entity& other);
};

#endif // ENTITY_H
