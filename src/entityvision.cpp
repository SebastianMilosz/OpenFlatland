#include "entityvision.hpp"
#include "physicsbody.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVision::EntityVision( codeframe::ObjectNode* parent ) :
    Object( "Vision", parent ),
    CastRays         ( this, "CastRays"         , false               , cPropertyInfo().Kind( KIND_LOGIC  ).Description("CastRays") ),
    RaysCnt          ( this, "RaysCnt"          , 100U                , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysCnt") ),
    RaysSize         ( this, "RaysSize"         , 100U                , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysSize") ),
    RaysStartingAngle( this, "RaysStartingAngle", -45                 , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysStartingAngle"), this, nullptr, &EntityVision::SetRaysStartingAngle),
    RaysEndingAngle  ( this, "RaysEndingAngle"  ,  45                 , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysEndingAngle"), this, nullptr, &EntityVision::SetRaysEndingAngle),
    VisionVector     ( this, "VisionVector"     , std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).ReferencePath("../ANN/AnnLayer[0].Input").Description("VisionVector"), this, &EntityVision::GetDistanceVector ),
    FixtureVector    ( this, "FixtureVector"    , std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).ReferencePath("../ANN/AnnLayer[1].Input").Description("FixtureVector"), this, &EntityVision::GetFixtureVector ),
    m_visionShape()
{
    m_rayLine[0].color = sf::Color::White;
    m_rayLine[1].color = sf::Color::White;

    m_visionShape.setRadius( PhysicsBody::sDescriptor::PIXELS_IN_METER * 0.6f);
    m_visionShape.setOutlineThickness(1);
    m_visionShape.setOrigin(15.0F, 15.0F);
    m_visionShape.setPointCount(16);
    m_visionShape.setStartAngle( -45 );
    m_visionShape.setEndAngle( 45 );

    m_visionShape.setOutlineColor( sf::Color::White );
    m_visionShape.setFillColor( sf::Color::Transparent );

#ifdef ENTITY_VISION_DEBUG
    m_directionRayLine[0].color = sf::Color::Red;
    m_directionRayLine[1].color = sf::Color::Red;
#endif // ENTITY_VISION_DEBUG
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVision::EntityVision( const EntityVision& other ) :
    Object( other ),
    CastRays( other.CastRays ),
    RaysCnt ( other.RaysCnt ),
    RaysSize( other.RaysSize ),
    RaysStartingAngle( other.RaysStartingAngle ),
    RaysEndingAngle( other.RaysEndingAngle ),
    VisionVector( other.VisionVector ),
    FixtureVector( other.FixtureVector )
{
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
EntityVision::sRay::sRay() :
    P1( 0.0F, 0.0F ),
    P2( 0.0F, 0.0F ),
    Fixture( 0.0F )
{

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
void EntityVision::draw( sf::RenderTarget& target, sf::RenderStates states ) const
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::Draw( sf::RenderWindow& window )
{
    // Drawing rays if configured
    if ( (bool)CastRays == true )
    {
        for ( auto it = m_visionVector.begin(); it != m_visionVector.end(); ++it )
        {
#ifdef ENTITY_VISION_DEBUG
            if ( it == m_visionVector.begin() )
            {
                m_rayLine[0].color = sf::Color::Blue;
                m_rayLine[1].color = sf::Color::Blue;
            }
            else
            {
                m_rayLine[0].color = sf::Color::White;
                m_rayLine[1].color = sf::Color::White;
            }
#endif // ENTITY_VISION_DEBUG

            m_rayLine[0].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( it->P1 );
            m_rayLine[1].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( it->P2 );

            window.draw( m_rayLine, 2, sf::Lines );
        }

#ifdef ENTITY_VISION_DEBUG
        m_directionRayLine[0].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( m_directionRay.P1 );
        m_directionRayLine[1].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( m_directionRay.P2 );
        window.draw( m_directionRayLine, 2, sf::Lines );
#endif // ENTITY_VISION_DEBUG
    }

    m_visionShape.setOutlineColor( GetDistanceVector() );

    m_visionShape.setPosition( m_x, m_y );
    m_visionShape.setRotation( m_r );

    window.draw( m_visionShape );
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
    m_visionVector.emplace_back( ray );
    m_distanceVisionVector.emplace_back( (ray.P2-ray.P1).Length() );
    m_fixtureVisionVector.emplace_back( ray.Fixture );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
#ifdef ENTITY_VISION_DEBUG
void EntityVision::AddDirectionRay( EntityVision::sRay ray )
{
    m_directionRay = ray;
}
#endif // ENTITY_VISION_DEBUG

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

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::SetRaysStartingAngle( int value )
{
    m_visionShape.setStartAngle( value );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::SetRaysEndingAngle( int value )
{
    m_visionShape.setEndAngle( value );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setPosition(float x, float y)
{
    m_x = x;
    m_y = y;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setRotation(float angle)
{
    m_r = angle;
}
