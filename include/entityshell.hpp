#ifndef ENTITYSHELL_HPP
#define ENTITYSHELL_HPP

#include "physicsbody.hpp"

class EntityShell : public PhysicsBody
{
    public:
        std::string Role()            const { return "Object";      }
        std::string Class()           const { return "EntityShell"; }
        std::string BuildType()       const { return "Dynamic";     }
        std::string ConstructPatern() const { return "X,Y"; }

    public:
                 EntityShell( std::string name, int x, int y, int z );
                 EntityShell( const EntityShell& other );
        virtual ~EntityShell();

        codeframe::Property<int,         EntityShell> X;
        codeframe::Property<int,         EntityShell> Y;
        codeframe::Property<bool,        EntityShell> CastRays;
        codeframe::Property<std::string, EntityShell> Name;

        EntityShell& operator=(const EntityShell& other);

        virtual void Draw( sf::RenderWindow& window, b2Body* body );

        int GetX();
        void SetX(int val);
        int GetY();
        void SetY(int val);
        int GetZ();
        void SetZ(int val);

    private:

};

#endif // ENTITYSHELL_HPP
