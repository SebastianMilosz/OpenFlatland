#ifndef PHYSICSBODY_H
#define PHYSICSBODY_H

#include <serializable.h>
#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>

class PhysicsBody : public codeframe::cSerializable
{
    public:
        std::string Role()            const { return "Object";      }
        std::string Class()           const { return "PhysicsBody"; }
        std::string BuildType()       const { return "Dynamic";     }
        std::string ConstructPatern() const { return ""; }

    public:
        struct sEntityShellDescriptor
        {
            sEntityShellDescriptor() :
                Body( NULL ),
                Color( sf::Color::Red )
            {
            }

            b2Body*         Body;
            b2CircleShape   Shape;
            b2FixtureDef    FixtureDef;
            b2BodyDef       BodyDef;
            sf::Color       Color;
        };

        PhysicsBody( std::string name, codeframe::cSerializableInterface* parent );
        virtual ~PhysicsBody();
        PhysicsBody(const PhysicsBody& other);
        PhysicsBody& operator=(const PhysicsBody& other);

        void       SetColor( const sf::Color& color );
        sf::Color& GetColor();

        sEntityShellDescriptor& GetDescriptor();

    protected:

    private:
        sEntityShellDescriptor m_descryptor;
};

#endif // PHYSICSBODY_H
