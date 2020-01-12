#ifndef PHYSICSBODY_H
#define PHYSICSBODY_H

#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>
#include <serializable_object.hpp>

class PhysicsBody : public codeframe::Object
{
    CODEFRAME_META_CLASS_NAME( "PhysicsBody" );
    CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        struct sDescriptor
        {
            static const float PIXELS_IN_METER;
            static const float METER_IN_PIXELS;

            sDescriptor() :
                Body( NULL ),
                Shape( NULL ),
                Color( sf::Color::White )
            {
            }

            b2Body*         Body;
            b2Shape*        Shape;
            b2FixtureDef    FixtureDef;
            b2BodyDef       BodyDef;
            sf::Color       Color;

            static b2Vec2 Meters2Pixels( const b2Vec2& point )
            {
                b2Vec2 retpoint = point;
                retpoint *= PIXELS_IN_METER;
                return retpoint;
            }

            static sf::Vector2f Meters2SFMLPixels( const b2Vec2& point )
            {
                b2Vec2 retpoint = point;
                retpoint *= PIXELS_IN_METER;

                return sf::Vector2f(
                                        retpoint.x,
                                        retpoint.y
                                   );
            }

            static b2Vec2 Pixels2Meters( const b2Vec2& point )
            {
                b2Vec2 retpoint = point;
                retpoint *= METER_IN_PIXELS;
                return retpoint;
            }
        };

        PhysicsBody( const std::string& name, codeframe::ObjectNode* parent );
        virtual ~PhysicsBody();
        PhysicsBody(const PhysicsBody& other);
        PhysicsBody& operator=(const PhysicsBody& other);

        void       SetColor( const sf::Color& color );
        sf::Color& GetColor();

        virtual void synchronize( b2Body& body ) = 0;

        sDescriptor& GetDescriptor();

    protected:

    private:
        sDescriptor m_descryptor;
};

#endif // PHYSICSBODY_H
