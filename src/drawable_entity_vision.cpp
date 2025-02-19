#include "drawable_entity_vision.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
DrawableEntityVision:: DrawableEntityVision(codeframe::ObjectNode* parent) :
    EntityVision(parent),
    ColorizeMode(this, "ColorizeMode", 0U, cPropertyInfo()
                 .Kind(KIND_ENUM)
                 .Enum("IronBow,RedYellow,BlueRed,BlackRed,BlueRedBin,BlueGreenRed,Grayscale,ShiftGray")
                 .Description("ColorizeMode"), nullptr, std::bind(&sf::ColorizeCircleShape::setColorizeMode, &m_visionShape, std::placeholders::_1)),
    m_visionShape(PhysicsBody::sDescriptor::PIXELS_IN_METER * 0.6f, 16),
    m_rayLines(2U * (unsigned int)RaysCnt)
{
    m_visionShape.setOutlineThickness(1);
    m_visionShape.setOrigin({15.0F, 15.0F});

    m_visionShape.setStartAngle(-45);
    m_visionShape.setEndAngle(45);

    m_visionShape.setOutlineColor(sf::Color::White);
    m_visionShape.setFillColor(sf::Color::Transparent);

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
DrawableEntityVision::DrawableEntityVision(const DrawableEntityVision& other) :
    EntityVision(other),
    ColorizeMode(other.ColorizeMode),
    m_rayLines(other.m_rayLines)
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::setPosition(sf::Vector2f position)
{
    EntityVision::setPosition(position);
    m_visionShape.setPosition(position);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::setRotation(sf::Angle angle)
{
    EntityVision::setRotation(angle);
    m_visionShape.setRotation(angle);
}


/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    // Drawing rays if configured
    if ((bool)DrawRays == true)
    {
        target.draw(m_rayLines.data(), m_rayLines.size(), sf::PrimitiveType::Lines);

#ifdef ENTITY_VISION_DEBUG
        target.draw(m_directionRayLine, 2U, sf::PrimitiveType::Lines);
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
void DrawableEntityVision::AddDirectionRay( EntityVision::Ray ray )
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
    EntityVision::EndFrame();
    m_visionShape.setOutlineColor( GetVisionVector() );
    PrepareRays();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::SetRaysStartingAngle( const int value )
{
    EntityVision::SetRaysStartingAngle(value);
    m_visionShape.setStartAngle(value);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::SetRaysEndingAngle( const int value )
{
    EntityVision::SetRaysEndingAngle(value);
    m_visionShape.setEndAngle(value);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::SetRaysCnt( const unsigned int cnt )
{
    m_rayLines.resize( 2U * cnt );

    PrepareRays();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void DrawableEntityVision::PrepareRays()
{
    // Drawing rays if configured
    if ( (bool)DrawRays == true )
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
