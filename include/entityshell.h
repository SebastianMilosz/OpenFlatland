#ifndef ENTITYSHELL_H
#define ENTITYSHELL_H

#include <serializable.h>
#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>

class EntityShell : public codeframe::cSerializable
{
    public:
        std::string Role()      const { return "Object";      }
        std::string Class()     const { return "EntityShell"; }
        std::string BuildType() const { return "Dynamic";     }

    public:
        struct sEntityShellDescriptor
        {
            sEntityShellDescriptor() :
                Body( NULL ),
                Color(sf::Color::Red)
            {
            }

            b2Body*         Body;
            b2CircleShape   Shape;
            b2FixtureDef    FixtureDef;
            b2BodyDef       BodyDef;
            sf::Color       Color;
        };

    public:
                 EntityShell( std::string name, int x, int y, int z );
                 EntityShell( const EntityShell& other );
        virtual ~EntityShell();

        EntityShell& operator=(const EntityShell& other);

        unsigned int GetX();
        void SetX(unsigned int val);
        unsigned int GetY();
        void SetY(unsigned int val);
        unsigned int GetZ();
        void SetZ(unsigned int val);

        void SetColor( const sf::Color& color ) { m_descryptor.Color = color; }
        sf::Color& GetColor() { return m_descryptor.Color; }

        sEntityShellDescriptor& GetDescriptor();

    private:
        sEntityShellDescriptor m_descryptor;
};

#endif // ENTITYSHELL_H
