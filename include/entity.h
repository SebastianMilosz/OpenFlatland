#ifndef ENTITY_H
#define ENTITY_H

#include <entityshell.h>
#include <entityghost.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class Entity : public EntityGhost
{
    public:
        Entity( int x, int y, int z );
        virtual ~Entity();
        Entity(const Entity& other);
        Entity& operator=(const Entity& other);
};

#endif // ENTITY_H
