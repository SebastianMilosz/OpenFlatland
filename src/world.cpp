#include "world.hpp"
#include "performance_logger.hpp"
#include "performance_application_def.hpp"

#include <string>
#include <sstream>
#include <cmath>
#include <omp.h>

#include <utilities/LoggerUtilities.h>

class QueryCallback : public b2QueryCallback
{
public:
   QueryCallback( const b2Vec2& point )
   {
      m_point = point;
      m_fixture = nullptr;
   }

   bool ReportFixture( b2Fixture* fixture )
   {
      b2Body* body = fixture->GetBody();
      if (body->GetType() == b2_dynamicBody)
      {
         if ( fixture->TestPoint(m_point) )
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
World::World( const std::string& name, ObjectNode* parent ) :
    Object( name, parent ),
    m_GroundBody( nullptr ),
    m_MouseJoint( nullptr ),
    m_Gravity( 0.f, 0.f ),
    m_World( m_Gravity ),
    m_entitySelMode( false )
{
    b2BodyDef groundBodyDef;
    groundBodyDef.position.Set(0, 0); // bottom-left corner
    m_GroundBody = m_World.CreateBody( &groundBodyDef );

    m_JointDef.bodyA = m_GroundBody;
    m_JointDef.bodyB = nullptr;

    m_JointDef.target.Set( 0.5f, 24.f );

    m_JointDef.maxForce = 0.5f;

    m_JointDef.frequencyHz = 4.0f;
    m_JointDef.dampingRatio = 0.5f;

    PERFORMANCE_ADD( PERFORMANCE_BOX2D_ONLY_PHYSIC_SYM, "Box2d physic" );
    PERFORMANCE_ADD( PERFORMANCE_BOX2D_RAYS_CAST,       "Box2d rays" );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::AddShell( smart_ptr<Entity> entity )
{
    PhysicsBody::sDescriptor& desc = entity->GetDescriptor();

    b2Body* body = m_World.CreateBody( &desc.BodyDef );

    if ( (b2Body*)nullptr != body )
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
void World::AddConst( smart_ptr<ConstElement> constElement )
{
    PhysicsBody::sDescriptor& desc = constElement->GetDescriptor();

    b2Body* body = m_World.CreateBody( &desc.BodyDef );

    if ( (b2Body*)nullptr != body )
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
bool World::PhysisStep(sf::RenderWindow& window)
{
    PERFORMANCE_ENTER( PERFORMANCE_BOX2D_ONLY_PHYSIC_SYM );

    m_World.Step(1/60.f, 8, 3);

    PERFORMANCE_LEAVE( PERFORMANCE_BOX2D_ONLY_PHYSIC_SYM );

    PERFORMANCE_ENTER( PERFORMANCE_BOX2D_RAYS_CAST );

    CalculateRays();

    PERFORMANCE_LEAVE( PERFORMANCE_BOX2D_RAYS_CAST );

    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    for ( const b2Body* BodyIterator = m_World.GetBodyList(); BodyIterator != nullptr; BodyIterator = BodyIterator->GetNext() )
    {
        const PhysicsBody* physicsBody = static_cast<const PhysicsBody*>( BodyIterator->GetUserData() );

        if ( (PhysicsBody*)nullptr != physicsBody )
        {
            const DrawableObject* drawableBody = dynamic_cast<const DrawableObject*>(physicsBody);
            if ((DrawableObject*)nullptr != drawableBody)
            {
                drawableBody->draw(target, sf::RenderStates::Default);
            }
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::synchronize()
{
    for ( b2Body* BodyIterator = m_World.GetBodyList(); BodyIterator != nullptr; BodyIterator = BodyIterator->GetNext() )
    {
        PhysicsBody*  physicsBody  = static_cast<PhysicsBody*>( BodyIterator->GetUserData() );

        if ( (PhysicsBody*)nullptr != physicsBody )
        {
            physicsBody->synchronize(*BodyIterator);
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::MouseDown(const float x, const float y)
{
    if( m_entitySelMode == false )
    {
        m_entitySelMode = true;

        b2Body* body = GetBodyAtMouse( x/PhysicsBody::sDescriptor::PIXELS_IN_METER, y/PhysicsBody::sDescriptor::PIXELS_IN_METER );

        if ( (b2Body*)nullptr != body )
        {
            EntityShell* entShell = static_cast<EntityShell*>( body->GetUserData() );

            if ( (EntityShell*)nullptr != entShell )
            {
                if ( m_MouseJoint )
                {
                    m_World.DestroyJoint( m_MouseJoint );
                    m_MouseJoint = nullptr;
                }

                b2Vec2 locationWorld = b2Vec2(x/PhysicsBody::sDescriptor::PIXELS_IN_METER, y/PhysicsBody::sDescriptor::PIXELS_IN_METER);

                m_JointDef.bodyB    = body;
                m_JointDef.maxForce = 1000.0f * body->GetMass();
                m_JointDef.target   = locationWorld;
                m_MouseJoint        = (b2MouseJoint*)m_World.CreateJoint( &m_JointDef );

                entShell->Selection().Select();

                body->SetAwake( true );
            }
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::MouseUp(const float x, const float y)
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
void World::MouseMove(const float x, const float y)
{
    if ( m_MouseJoint )
    {
        b2Vec2 locationWorld = b2Vec2(x/PhysicsBody::sDescriptor::PIXELS_IN_METER, y/PhysicsBody::sDescriptor::PIXELS_IN_METER);
        m_MouseJoint->SetTarget( locationWorld );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
b2Body* World::GetBodyAtMouse(const float x, const float y)
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

   if ( callback.m_fixture )
   {
        return callback.m_fixture->GetBody();
   }

   return nullptr;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void World::CalculateRays()
{
    for ( const b2Body* BodyIterator = m_World.GetBodyList(); BodyIterator != nullptr; BodyIterator = BodyIterator->GetNext() )
    {
        if ( BodyIterator->GetType() == b2_dynamicBody )
        {
            Entity* entity = static_cast<Entity*>( BodyIterator->GetUserData() );

            if ( (Entity*)nullptr != entity )
            {
                entity->Vision().CastRays(m_World, entity->GetPhysicalPoint());
            }
        }
    }
}
