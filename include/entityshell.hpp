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
        codeframe::Property<float,        EntityShell> Rotation;
        codeframe::Property<bool,         EntityShell> CastRays;
        codeframe::Property<unsigned int, EntityShell> RaysCnt;
        codeframe::Property<unsigned int, EntityShell> RaysSize;
        codeframe::Property<std::string,  EntityShell> Name;
        codeframe::Property<float,        EntityShell> Density;
        codeframe::Property<float,        EntityShell> Friction;

        EntityShell& operator=(const EntityShell& other);

        virtual void Draw( sf::RenderWindow& window, b2Body* body );

        int GetX();
        float32 GetPhysicalX();
        void SetX(int val);
        int GetY();
        float32 GetPhysicalY();
        void SetY(int val);

        float32 GetRotation();

        const b2Vec2& GetPhysicalPoint();

    private:
        b2Vec2 m_zeroVector;
};

#endif // ENTITYSHELL_HPP
