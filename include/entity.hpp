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
    public:
        std::string Role()      const { return "Object";  }
        std::string Class()     const { return "Entity";  }
        std::string BuildType() const { return "Dynamic"; }

    public:
        Entity( std::string name, int x, int y );
        virtual ~Entity();
        Entity(const Entity& other);
        Entity& operator=(const Entity& other);
};

#endif // ENTITY_H
