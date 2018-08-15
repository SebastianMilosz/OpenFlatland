#ifndef ENTITYSHELL_H
#define ENTITYSHELL_H

#include "physicsbody.h"

class EntityShell : public PhysicsBody
{
    public:
        std::string Role()            const { return "Object";      }
        std::string Class()           const { return "EntityShell"; }
        std::string BuildType()       const { return "Dynamic";     }
        std::string ConstructPatern() const { return "X,Y,Z"; }

    public:
                 EntityShell( std::string name, int x, int y, int z );
                 EntityShell( const EntityShell& other );
        virtual ~EntityShell();

        codeframe::Property<unsigned int, EntityShell> X;
        codeframe::Property<unsigned int, EntityShell> Y;
        codeframe::Property<unsigned int, EntityShell> Z;

        codeframe::Property<std::string, EntityShell> Name;

        EntityShell& operator=(const EntityShell& other);

        virtual void Draw( sf::RenderWindow& window, b2Body* body );

        unsigned int GetX();
        void SetX(unsigned int val);
        unsigned int GetY();
        void SetY(unsigned int val);
        unsigned int GetZ();
        void SetZ(unsigned int val);

    private:

};

#endif // ENTITYSHELL_H
