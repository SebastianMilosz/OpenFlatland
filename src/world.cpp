#include "world.h"

#include <string>
#include <sstream>

#include <utilities/LoggerUtilities.h>

// Box2D works with meters where 1mt = 30 pixels
static const float SCALE = 30.f;

namespace std {
    template<typename T>
    std::string to_string(const T &n) {
        std::ostringstream s;
        s << n;
        return s.str();
    }
}

class QueryCallback : public b2QueryCallback
{
public:
   QueryCallback(const b2Vec2& point)
   {
      m_point = point;
      m_fixture = NULL;
   }

   bool ReportFixture(b2Fixture* fixture)
   {
      b2Body* body = fixture->GetBody();
      if (body->GetType() == b2_dynamicBody)
      {
         bool inside = fixture->TestPoint(m_point);
         if (inside)
         {
            m_fixture = fixture;

            // We are done, terminate the query.
            return false;
         }
      }
      // Continue the query.
      return true;
   }
   b2Vec2 m_point;
   b2Fixture* m_fixture;
};

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
World::World() :
    m_Gravity( 0.f, 0.f ),
    m_World( m_Gravity )
{
     // Load it from a file
     if (!m_font.loadFromFile("arial.ttf"))
     {
         // error...
     }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
World::~World()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool World::AddShell( EntityShell& shell )
{
    EntityShell::sEntityShellDescriptor& desc = shell.GetDescriptor();

    desc.Body = m_World.CreateBody( &desc.BodyDef );
    desc.Body->CreateFixture( &desc.FixtureDef );

    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool World::PhysisStep()
{
    m_World.Step(1/60.f, 8, 3);
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool World::Draw( sf::RenderWindow& window )
{
    for( b2Body* BodyIterator = m_World.GetBodyList(); BodyIterator != 0; BodyIterator = BodyIterator->GetNext() )
    {
        if (BodyIterator->GetType() == b2_dynamicBody)
        {
            void* userVoid = BodyIterator->GetUserData();

            if( userVoid )
            {
                EntityShell* entShell = static_cast<EntityShell*>(userVoid);

                if( entShell )
                {
                    sf::Color& entColor = entShell->GetColor();

                    float xpos = BodyIterator->GetPosition().x;
                    float ypos = BodyIterator->GetPosition().y;

                    sf::CircleShape circle;
                    circle.setRadius(10);
                    circle.setOutlineColor( entColor );
                    circle.setOutlineThickness(1);
                    circle.setOrigin(16.f, 16.f);
                    circle.setPosition(SCALE * xpos, SCALE * ypos);
                    circle.setRotation(BodyIterator->GetAngle() * 180/b2_pi);
                    window.draw(circle);

                    sf::Text text;
                    text.setString( std::string("(") + std::to_string(xpos) + std::string(", ") + std::to_string(ypos) + std::string(")") );
                    text.setColor(sf::Color::White);
                    text.setCharacterSize(12);
                    text.setFont(m_font);
                    text.setPosition(SCALE * xpos, SCALE * ypos);
                    window.draw(text);
                }
            }
        }
        else
        {
            sf::Sprite GroundSprite;
            //GroundSprite.SetTexture(GroundTexture);
            GroundSprite.setOrigin(400.f, 8.f);
            GroundSprite.setPosition(BodyIterator->GetPosition().x * SCALE, BodyIterator->GetPosition().y * SCALE);
            GroundSprite.setRotation(180/b2_pi * BodyIterator->GetAngle());
            window.draw(GroundSprite);
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool World::MouseDown( float x, float y )
{
    b2Body* body = getBodyAtMouse( x/SCALE, y/SCALE );

    if( body )
    {
        void* userVoid = body->GetUserData();

        if( userVoid )
        {
            EntityShell* entShell = static_cast<EntityShell*>(userVoid);

            if( entShell )
            {
                LOGGER( LOG_INFO << "Entity Selected: x=" << entShell->GetX() << " x=" << entShell->GetY() );

                entShell->SetColor( sf::Color::Blue );
                return true;
            }
        }
    }

    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
b2Body* World::getBodyAtMouse( float x, float y )
{
   b2Vec2 mouseV2;
   mouseV2.Set(x,y);

   // small box:
   b2AABB aabb = b2AABB();
   aabb.lowerBound.Set(x -0.001, y - 0.001);
   aabb.upperBound.Set(x +0.001, y + 0.001);

   // Query the world for overlapping shapes.
   QueryCallback callback(mouseV2);
   m_World.QueryAABB(&callback, aabb);

   if (callback.m_fixture)
   {
        return callback.m_fixture->GetBody();
   }

   return NULL;
}
