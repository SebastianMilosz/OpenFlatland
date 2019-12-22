#include "drawable_entity_vision.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableEntityVision:: DrawableEntityVision( codeframe::ObjectNode* parent ) :
    EntityVision( parent ),
    m_visionShape( PhysicsBody::sDescriptor::PIXELS_IN_METER * 0.6f, 16 ),
    m_rayLines( 2U * (unsigned int)RaysCnt )
{
    m_visionShape.setOutlineThickness(1);
    m_visionShape.setOrigin(15.0F, 15.0F);

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
DrawableEntityVision::DrawableEntityVision( const DrawableEntityVision& other ) :
    EntityVision( other ),
    m_rayLines( other.m_rayLines )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableEntityVision::~ DrawableEntityVision()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::setPosition(float x, float y)
{
    EntityVisionNode::setPosition( x, y );
    m_visionShape.setPosition( x, y );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::setRotation(float angle)
{
    EntityVisionNode::setRotation( angle );
    m_visionShape.setRotation( angle );
}


/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::draw( sf::RenderTarget& target, sf::RenderStates states ) const
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
#ifdef ENTITY_VISION_DEBUG
void DrawableEntityVision::AddDirectionRay( EntityVision::sRay ray )
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
void DrawableEntityVision::EndFrame()
{
    m_visionShape.setOutlineColor( GetDistanceVector() );
    //PrepareRays();
}
