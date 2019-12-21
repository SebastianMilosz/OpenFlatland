#include "entity_vision.hpp"
#include "physics_body.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVision::EntityVision( codeframe::ObjectNode* parent ) :
    Object( "Vision", parent ),
    CastRays         ( this, "CastRays"         , false               , cPropertyInfo().Kind( KIND_LOGIC  ).Description("CastRays") ),
    RaysCnt          ( this, "RaysCnt"          , 100U                , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysCnt"), this, nullptr, &EntityVision::SetRaysCnt ),
    RaysSize         ( this, "RaysSize"         , 100U                , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysSize") ),
    RaysStartingAngle( this, "RaysStartingAngle", -45                 , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysStartingAngle"), this, nullptr, &EntityVision::SetRaysStartingAngle),
    RaysEndingAngle  ( this, "RaysEndingAngle"  ,  45                 , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysEndingAngle"), this, nullptr, &EntityVision::SetRaysEndingAngle),
    VisionVector     ( this, "VisionVector"     , std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).ReferencePath("../ANN/AnnLayer[0].Input").Description("VisionVector"), this, &EntityVision::GetDistanceVector ),
    FixtureVector    ( this, "FixtureVector"    , std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).ReferencePath("../ANN/AnnLayer[1].Input").Description("FixtureVector"), this, &EntityVision::GetFixtureVector ),
    m_visionShape(),
    m_rayLines( 2U * (unsigned int)RaysCnt )
{
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
    FixtureVector( other.FixtureVector ),
    m_rayLines( other.m_rayLines )
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
void EntityVision::draw( sf::RenderTarget& target, sf::RenderStates states ) const
{
    // Drawing rays if configured
    if ( (bool)CastRays == true )
    {
        target.draw( m_rayLines.data(), m_rayLines.size(), sf::Lines );

#ifdef ENTITY_VISION_DEBUG
        target.draw( m_directionRayLine, 2U, sf::Lines );
#endif // ENTITY_VISION_DEBUG
    }

    target.draw( m_visionShape );
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
    m_directionRayLine[0].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( m_directionRay.P1 );
    m_directionRayLine[1].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( m_directionRay.P2 );
}
#endif // ENTITY_VISION_DEBUG

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setPosition(float x, float y)
{
    EntityVisionNode::setPosition( x, y );
    m_visionShape.setPosition( x, y );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setRotation(float angle)
{
    EntityVisionNode::setRotation( angle );
    m_visionShape.setRotation( angle );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::EndFrame()
{
    m_visionShape.setOutlineColor( GetDistanceVector() );
    PrepareRays();
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
void EntityVision::SetRaysCnt( unsigned int cnt )
{
    m_rayLines.resize( 2U * cnt );

    PrepareRays();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::PrepareRays()
{
    // Drawing rays if configured
    if ( (bool)CastRays == true )
    {
        size_t n = 0U;
        for ( auto it = m_visionVector.begin(); it != m_visionVector.end(); ++it )
        {
            m_rayLines[ n   ].color    = sf::Color::White;
            m_rayLines[ n++ ].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( it->P1 );
            m_rayLines[ n   ].color    = sf::Color::White;
            m_rayLines[ n++ ].position = PhysicsBody::sDescriptor::Meters2SFMLPixels( it->P2 );
        }

#ifdef ENTITY_VISION_DEBUG
        m_rayLines[ 0U ].color = sf::Color::Blue;
        m_rayLines[ 1U ].color = sf::Color::Blue;
#endif // ENTITY_VISION_DEBUG
    }
}
