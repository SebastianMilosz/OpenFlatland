#include "entityshell.hpp"

#include <utilities/LoggerUtilities.h>
#include <utilities/TextUtilities.h>
#include <utilities/MathUtilities.h>

#include "fontfactory.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell( const std::string& name, int x, int y ) :
    PhysicsBody( name, nullptr ),
    X                ( this, "X"                , 0    , cPropertyInfo().Kind( KIND_NUMBER ).Description("Xpos"), this, &EntityShell::GetX ),
    Y                ( this, "Y"                , 0    , cPropertyInfo().Kind( KIND_NUMBER ).Description("Ypos"), this, &EntityShell::GetY ),
    Rotation         ( this, "R"                , 0.0F , cPropertyInfo().Kind( KIND_REAL   ).Description("Rotation"), this, &EntityShell::GetRotation, &EntityShell::SetRotation ),
    CastRays         ( this, "CastRays"         , false, cPropertyInfo().Kind( KIND_LOGIC  ).Description("CastRays") ),
    RaysCnt          ( this, "RaysCnt"          , 100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysCnt") ),
    RaysSize         ( this, "RaysSize"         , 100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysSize") ),
    RaysStartingAngle( this, "RaysStartingAngle", -45  , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysStartingAngle"), this, nullptr, &EntityShell::SetRaysStartingAngle),
    RaysEndingAngle  ( this, "RaysEndingAngle"  ,  45  , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysEndingAngle"), this, nullptr, &EntityShell::SetRaysEndingAngle),
    Name             ( this, "Name"             , ""   , cPropertyInfo().Kind( KIND_TEXT   ).Description("Name") ),
    Density          ( this, "Density"          , 1.F  , cPropertyInfo().Kind( KIND_REAL   ).Description("Density") ),
    Friction         ( this, "Friction"         , 0.7F , cPropertyInfo().Kind( KIND_REAL   ).Description("Friction") ),
    m_zeroVector( 0.0F, 0.0F ),
    m_visionShape(),
    m_triangle( sDescriptor::PIXELS_IN_METER * 0.5f, 3 ),
    m_vision( this ),
    m_curX(0),
    m_curY(0),
    m_curR(0.0F)
{
    b2CircleShape* shape =  new b2CircleShape();
    shape->m_p.Set(0, 0);
    shape->m_radius = 15.0f/sDescriptor::PIXELS_IN_METER;

    GetDescriptor().Shape = shape;
    GetDescriptor().BodyDef.position = b2Vec2(
                                              (float)x/sDescriptor::PIXELS_IN_METER,
                                              (float)y/sDescriptor::PIXELS_IN_METER
                                             );
    GetDescriptor().BodyDef.type = b2_dynamicBody;
    GetDescriptor().BodyDef.userData    = (void*)this;
    GetDescriptor().FixtureDef.density  = (float)Density;
    GetDescriptor().FixtureDef.friction = (float)Friction;
    GetDescriptor().FixtureDef.shape    = GetDescriptor().Shape;

    m_visionShape.setRadius(sDescriptor::PIXELS_IN_METER * 0.6f);
    m_visionShape.setOutlineThickness(1);
    m_visionShape.setOrigin(15.0F, 15.0F);
    m_visionShape.setPointCount(16);
    m_visionShape.setStartAngle( -45 );
    m_visionShape.setEndAngle( 45 );

    m_triangle.setOutlineThickness(1);
    m_triangle.setOrigin(12.5F, 12.5F);
    m_triangle.setFillColor( sf::Color::Transparent );

    m_visionShape.setOutlineColor( GetColor() );
    m_visionShape.setFillColor( sf::Color::Transparent );

    Selection().signalSelectionChanged.connect( this, &EntityShell::slotSelectionChanged );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::~EntityShell()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::slotSelectionChanged( smart_ptr<ObjectNode> )
{
    if ( Selection().IsSelected() == true )
    {
        m_triangle.setFillColor( sf::Color::Blue );
    }
    else
    {
        m_triangle.setFillColor( sf::Color::Transparent );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell( const EntityShell& other ) :
    PhysicsBody( other ),
    X       ( other.X ),
    Y       ( other.Y ),
    Rotation( other.Rotation ),
    CastRays( other.CastRays ),
    RaysCnt ( other.RaysCnt ),
    RaysSize( other.RaysSize ),
    RaysStartingAngle( other.RaysStartingAngle ),
    RaysEndingAngle( other.RaysEndingAngle ),
    Name    ( other.Name ),
    Density ( other.Density ),
    Friction( other.Friction ),
    m_vision( other.m_vision )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell& EntityShell::operator=( const EntityShell& rhs )
{
    PhysicsBody::operator = (rhs);

    //assignment operator
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::Draw( sf::RenderWindow& window, b2Body* body )
{
    if( (b2Body*)nullptr != body )
    {
        float xpos( body->GetPosition().x * sDescriptor::PIXELS_IN_METER );
        float ypos( body->GetPosition().y * sDescriptor::PIXELS_IN_METER );
        float rot ( body->GetAngle() * 180.0F/b2_pi );

        // Drawing rays if configured
        if ( (bool)CastRays == true )
        {
            m_vision.Draw( window );
        }

        m_visionShape.setOutlineColor( m_vision.GetDistanceVector() );

        m_visionShape.setPosition( xpos, ypos );
        m_visionShape.setRotation( rot );

        m_triangle.setPosition( xpos, ypos );
        m_triangle.setRotation( rot );

        window.draw( m_visionShape );
        window.draw( m_triangle );
    }
}

/*****************************************************************************/
/**
  * @brief Return X coordinates in pixels
 **
******************************************************************************/
const int& EntityShell::GetX()
{
    if ( GetDescriptor().Body != nullptr )
    {
        m_curX = GetDescriptor().Body->GetPosition().x * sDescriptor::PIXELS_IN_METER;
    }
    return m_curX;
}

/*****************************************************************************/
/**
  * @brief Return X coordinates in meters
 **
******************************************************************************/
float32 EntityShell::GetPhysicalX()
{
    if ( GetDescriptor().Body == nullptr )
    {
        return 0.0F;
    }

    return GetDescriptor().Body->GetPosition().x;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetX(int val)
{
    //m_x = val;
}

/*****************************************************************************/
/**
  * @brief Return Y coordinates in pixels
 **
******************************************************************************/
const int& EntityShell::GetY()
{
    if ( GetDescriptor().Body != nullptr )
    {
        m_curY = GetDescriptor().Body->GetPosition().y * sDescriptor::PIXELS_IN_METER;
    }
    return m_curY;
}

/*****************************************************************************/
/**
  * @brief Return Y coordinates in meters
 **
******************************************************************************/
float32 EntityShell::GetPhysicalY()
{
    if ( GetDescriptor().Body == nullptr )
    {
        return 0.0F;
    }

    return GetDescriptor().Body->GetPosition().y;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetY(int val)
{
    //m_y = val;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetRaysStartingAngle( int value )
{
    m_visionShape.setStartAngle( value );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetRaysEndingAngle( int value )
{
    m_visionShape.setEndAngle( value );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const float32& EntityShell::GetRotation()
{
    if ( GetDescriptor().Body != nullptr )
    {
        static const float pi = 3.141592654F;

        m_curR = utilities::math::ConstrainAngle( GetDescriptor().Body->GetAngle() * (180.0/pi) );
    }
    return m_curR;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityShell::SetRotation( float rotation )
{
    b2Body* body = GetDescriptor().Body;

    if ( (b2Body*)nullptr != body )
    {
        static const float pi = 3.141592654F;

        float angleToSet( rotation / (180.0/pi) );

        body->SetTransform( body->GetPosition(), angleToSet );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const b2Vec2& EntityShell::GetPhysicalPoint()
{
    if ( GetDescriptor().Body == nullptr )
    {
        return m_zeroVector;
    }

    return GetDescriptor().Body->GetPosition();
}
