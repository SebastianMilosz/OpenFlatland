#include "entityvision.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVision::EntityVision( codeframe::cSerializableInterface* parent ) :
    cSerializable( "Vision", parent ),
    VisionVector ( this, "VisionVector" , std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).Description("VisionVector"), this, &EntityVision::GetDistanceVector ),
    FixtureVector( this, "FixtureVector", std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).Description("FixtureVector"), this, &EntityVision::GetFixtureVector )
{
    m_rayLine[0].color = sf::Color::White;
    m_rayLine[1].color = sf::Color::White;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVision::~EntityVision()
{
    m_visionVector.clear();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVision::sRay::sRay( b2Vec2& p1, b2Vec2& p2, float32 f ) :
    P1( p1 ),
    P2( p2 ),
    Fixture( f )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::Draw( sf::RenderWindow& window )
{
    for(std::vector<EntityVision::sRay>::iterator it = m_visionVector.begin(); it != m_visionVector.end(); ++it)
    {
        m_rayLine[0].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( it->P1 );
        m_rayLine[1].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( it->P2 );

        window.draw( m_rayLine, 2, sf::Lines );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::StartFrame()
{
    m_visionVector.clear();
    m_distanceVisionVector.clear();
    m_fixtureVisionVector.clear();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::AddRay( EntityVision::sRay ray )
{
    m_visionVector.push_back( ray );
    m_distanceVisionVector.push_back( (ray.P2-ray.P1).Length() );
    m_fixtureVisionVector.push_back( ray.Fixture );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::EndFrame()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const std::vector<float>& EntityVision::GetDistanceVector()
{
    return m_distanceVisionVector;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const std::vector<float>& EntityVision::GetFixtureVector()
{
    return m_fixtureVisionVector;
}
