#include "world.h"
#include "entityshell.h"

static const float SCALE = 30.f;

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
b2Body* World::CreateBody( b2BodyDef* def )
{
    return m_World.CreateBody( def );
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

                    sf::CircleShape circle;
                    circle.setRadius(10);
                    circle.setOutlineColor( entColor );
                    circle.setOutlineThickness(5);
                    circle.setOrigin(16.f, 16.f);
                    circle.setPosition(SCALE * BodyIterator->GetPosition().x, SCALE * BodyIterator->GetPosition().y);
                    circle.setRotation(BodyIterator->GetAngle() * 180/b2_pi);
                    window.draw(circle);

                    sf::Text text;
                    text.setString("Hello world");
                    text.setColor(sf::Color::White);
                    text.setCharacterSize(24);
                    text.setFont(m_font);
                    text.setPosition(SCALE * BodyIterator->GetPosition().x + 10, SCALE * BodyIterator->GetPosition().y + 10);
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
bool World::MouseDown( int x, int y )
{
    b2Body* body = getBodyAtMouse( x / SCALE, y / SCALE );

    if( body )
    {
        void* userVoid = body->GetUserData();

        if( userVoid )
        {
            EntityShell* entShell = static_cast<EntityShell*>(userVoid);

            if( entShell )
            {
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
b2Body* World::getBodyAtMouse( int x, int y )
{
   b2Vec2 mouseV2;
   mouseV2.Set(x,y);

   // small box:
   b2AABB aabb = b2AABB();
   aabb.lowerBound.Set(x -1.001, y - 1.001);
   aabb.upperBound.Set(x +1.001, y + 1.001);

   // Query the world for overlapping shapes.
   QueryCallback callback(mouseV2);
   m_World.QueryAABB(&callback, aabb);

   if (callback.m_fixture)
   {
        return callback.m_fixture->GetBody();
   }

   return NULL;
}
