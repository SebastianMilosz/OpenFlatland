#ifndef ENTITY_H
#define ENTITY_H

#include <serializable.h>

#include "entityshell.h"
#include "entityghost.h"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class Entity : public EntityGhost
{
    public:
        std::string Role()      const { return "Object";  }
        std::string Class()     const { return "Entity";  }
        std::string BuildType() const { return "Dynamic"; }

    public:
        Entity( std::string name, int x, int y, int z );
        virtual ~Entity();
        Entity(const Entity& other);
        Entity& operator=(const Entity& other);
};

#endif // ENTITY_H
