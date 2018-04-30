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
    m_World( m_Gravity ),
    m_MouseJoint( NULL ),
    m_GroundBody( NULL ),
    m_entitySelMode( false )
{
     // Load it from a file
     if (!m_font.loadFromFile("arial.ttf"))
     {
         // error...
     }

    b2BodyDef groundBodyDef;
    groundBodyDef.position.Set(0, 0); // bottom-left corner
    m_GroundBody = m_World.CreateBody( &groundBodyDef );

    m_JointDef.bodyA = m_GroundBody;
    m_JointDef.bodyB = NULL;

    m_JointDef.target.Set( 0.5f, 24.f );

    m_JointDef.maxForce = 0.5f;

    m_JointDef.frequencyHz = 4.0f;
    m_JointDef.dampingRatio = 0.5f;
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
bool World::AddShell( std::shared_ptr<EntityShell> shell )
{
    EntityShell::sEntityShellDescriptor& desc = shell->GetDescriptor();

    b2Body* body = m_World.CreateBody( &desc.BodyDef );

    if ( (b2Body*)NULL != body )
    {
        body->CreateFixture( &desc.FixtureDef );
        desc.Body = body;
        return true;
    }

    return false;
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
    for ( b2Body* BodyIterator = m_World.GetBodyList(); BodyIterator != 0; BodyIterator = BodyIterator->GetNext() )
    {
        if (BodyIterator->GetType() == b2_dynamicBody)
        {
            void* userVoid = BodyIterator->GetUserData();

            if ( userVoid )
            {
                EntityShell* entShell = static_cast<EntityShell*>(userVoid);

                if ( entShell )
                {
                    sf::Color& entColor = entShell->GetColor();

                    float xpos = BodyIterator->GetPosition().x;
                    float ypos = BodyIterator->GetPosition().y;

                    sf::CircleShape circle;
                    circle.setRadius(SCALE * 0.5f);
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
void World::MouseDown( float x, float y )
{
    if( m_entitySelMode == false )
    {
        m_entitySelMode = true;

        b2Body* body = getBodyAtMouse( x/SCALE, y/SCALE );

        if ( body )
        {
            void* userVoid = body->GetUserData();

            if ( userVoid )
            {
                EntityShell* entShell = static_cast<EntityShell*>(userVoid);

                if ( (EntityShell*)NULL != entShell )
                {
                    b2Body* body = entShell->GetDescriptor().Body;

                    if ( body )
                    {
                        if ( m_MouseJoint )
                        {
                            m_World.DestroyJoint( m_MouseJoint );
                            m_MouseJoint = NULL;
                        }

                        b2Vec2 locationWorld = b2Vec2(x/SCALE, y/SCALE);

                        m_JointDef.bodyB    = body;
                        m_JointDef.maxForce = 1000.0f * body->GetMass();
                        m_JointDef.target   = locationWorld;
                        m_MouseJoint        = (b2MouseJoint*)m_World.CreateJoint( &m_JointDef );

                        entShell->SetColor( sf::Color::Blue );
                        body->SetAwake(true);
                    }
                }
            }
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::MouseUp( float x, float y )
{
    m_entitySelMode = false;

	if ( m_MouseJoint )
	{
		m_World.DestroyJoint( m_MouseJoint );
		m_MouseJoint = NULL;
	}
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::MouseMove( float x, float y )
{
	if ( m_MouseJoint )
	{
	    b2Vec2 locationWorld = b2Vec2(x/SCALE, y/SCALE);
		m_MouseJoint->SetTarget( locationWorld );
	}
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
