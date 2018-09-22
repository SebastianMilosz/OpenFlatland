#include "entityvision.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVision::EntityVision( codeframe::cSerializableInterface* parent ) :
    cSerializable( "Vision", parent ),
    VisionVector( this, "VisionVector", std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).Description("VisionVector"), this, &EntityVision::GetRay )
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
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::AddRay( EntityVision::sRay ray )
{
    m_visionVector.push_back( ray );
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
const std::vector<float>& EntityVision::GetRay()
{
    return m_distanceVisionVector;
}
