#include "entity.h"

Entity::Entity()
{
    //ctor
}

Entity::~Entity()
{
    //dtor
}

Entity::Entity(const Entity& other)
{
    //copy ctor
}

Entity& Entity::operator=(const Entity& rhs)
{
    if (this == &rhs) return *this; // handle self assignment
    //assignment operator
    return *this;
}
