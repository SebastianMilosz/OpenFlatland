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
                 EntityShell( std::string name, int x, int y );
                 EntityShell( const EntityShell& other );
        virtual ~EntityShell();

        codeframe::Property<int,          EntityShell> X;
        codeframe::Property<int,          EntityShell> Y;
        codeframe::Property<bool,         EntityShell> CastRays;
        codeframe::Property<unsigned int, EntityShell> RaysCnt;
        codeframe::Property<std::string,  EntityShell> Name;
        codeframe::Property<float,        EntityShell> Density;
        codeframe::Property<float,        EntityShell> Friction;

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
