#ifndef ENTITY_H
#define ENTITY_H

#include <entityshell.h>

class Entity
{
    public:
        Entity();
        virtual ~Entity();
        Entity(const Entity& other);
        Entity& operator=(const Entity& other);

        void AddShell( std::shared_ptr<EntityShell> shell )
        {
            m_shell = shell;
        }

    protected:

    private:
        std::shared_ptr<EntityShell> m_shell;
};

#endif // ENTITY_H
