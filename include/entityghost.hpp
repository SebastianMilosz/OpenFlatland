#ifndef ENTITYGHOST_HPP
#define ENTITYGHOST_HPP

#include "entityshell.hpp"

class EntityGhost : public EntityShell
{
    public:
        std::string Role()      const { return "Object";      }
        std::string Class()     const { return "EntityGhost"; }
        std::string BuildType() const { return "Dynamic";     }

    public:
        EntityGhost( std::string name, int x, int y, int z );
        virtual ~EntityGhost();
        EntityGhost(const EntityGhost& other);
        EntityGhost& operator=(const EntityGhost& other);

    protected:

    private:
};

#endif // ENTITYGHOST_HPP