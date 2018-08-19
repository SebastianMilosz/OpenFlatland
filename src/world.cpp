#include "world.hpp"

#include <string>
#include <sstream>

#include <utilities/LoggerUtilities.h>

// Box2D works with meters where 1mt = 30 pixels
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
World::World( std::string name, cSerializableInterface* parent ) :
    cSerializable( name, parent ),
    m_Gravity( 0.f, 0.f ),
    m_World( m_Gravity ),
    m_MouseJoint( NULL ),
    m_GroundBody( NULL ),
    m_entitySelMode( false )
{
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
void World::AddShell( std::shared_ptr<Entity> entity )
{
    PhysicsBody::sDescriptor& desc = entity->GetDescriptor();

    b2Body* body = m_World.CreateBody( &desc.BodyDef );

    if ( (b2Body*)NULL != body )
    {
        body->CreateFixture( &desc.FixtureDef );
        desc.Body = body;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::AddConst( std::shared_ptr<ConstElement> constElement )
{
    PhysicsBody::sDescriptor& desc = constElement->GetDescriptor();

    b2Body* body = m_World.CreateBody( &desc.BodyDef );

    if ( (b2Body*)NULL != body )
    {
        body->CreateFixture( &desc.FixtureDef );
        desc.Body = body;
    }
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
        if ( BodyIterator->GetType() == b2_dynamicBody ||
             BodyIterator->GetType() == b2_staticBody )
        {
            void* userVoid = BodyIterator->GetUserData();

            if ( userVoid )
            {
                PhysicsBody* physicsBody = static_cast<PhysicsBody*>(userVoid);

                if ( physicsBody )
                {
                    physicsBody->Draw( window, BodyIterator );
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

    return true;
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

                    entShell->Select();

                    body->SetAwake( true );
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
