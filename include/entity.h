#ifndef ENTITY_H
#define ENTITY_H

#include <entityshell.h>
#include <entityghost.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class Entity
{
    public:
        Entity();
        virtual ~Entity();
        Entity(const Entity& other);
        Entity& operator=(const Entity& other);

        void AddShell( std::shared_ptr<EntityShell> shell );
        void AddGhost( std::shared_ptr<EntityGhost> ghost );

    protected:

    private:
        std::shared_ptr<EntityShell> m_shell;
        std::shared_ptr<EntityGhost> m_ghost;
};

#endif // ENTITY_H
