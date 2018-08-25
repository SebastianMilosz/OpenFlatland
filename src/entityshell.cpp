#include "entityshell.hpp"

#include <utilities/LoggerUtilities.h>
#include <utilities/TextUtilities.h>

#include "fontfactory.hpp"

using namespace codeframe;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell::EntityShell( std::string name, int x, int y ) :
    PhysicsBody( name, NULL ),
    X       ( this, "X"       , 0    , cPropertyInfo().Kind( KIND_NUMBER ).Description("Xpos"), this, &EntityShell::GetX ),
    Y       ( this, "Y"       , 0    , cPropertyInfo().Kind( KIND_NUMBER ).Description("Ypos"), this, &EntityShell::GetY ),
    Rotation( this, "R"       , 0.0F , cPropertyInfo().Kind( KIND_REAL   ).Description("Rotation"), this, &EntityShell::GetRotation ),
    CastRays( this, "CastRays", false, cPropertyInfo().Kind( KIND_LOGIC  ).Description("CastRays") ),
    RaysCnt ( this, "RaysCnt" , 100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysCnt") ),
    RaysSize( this, "RaysSize", 100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysSize") ),
    Name    ( this, "Name"    , ""   , cPropertyInfo().Kind( KIND_TEXT   ).Description("Name") ),
    Density ( this, "Density" , 1.F  , cPropertyInfo().Kind( KIND_REAL   ).Description("Density") ),
    Friction( this, "Friction", 0.7F , cPropertyInfo().Kind( KIND_REAL   ).Description("Friction") ),
    m_triangle( sDescriptor::PIXELS_IN_METER * 0.5f, 3 ),
    m_zeroVector( 0.0F, 0.0F )
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

    m_circle.setRadius(sDescriptor::PIXELS_IN_METER * 0.5f);
    m_circle.setOutlineThickness(2);
    m_circle.setOrigin(12.5F, 12.5F);
    m_circle.setPointCount(16);

    m_triangle.setOutlineThickness(1);
    m_triangle.setOrigin(12.5F, 12.5F);
    m_triangle.setFillColor( sf::Color::Transparent );
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
EntityShell::EntityShell(const EntityShell& other) :
    PhysicsBody( other ),
    X       ( other.X ),
    Y       ( other.Y ),
    Rotation( other.Rotation ),
    CastRays( other.CastRays ),
    RaysCnt ( other.RaysCnt ),
    RaysSize( other.RaysSize ),
    Name    ( other.Name ),
    Density ( other.Density ),
    Friction( other.Friction )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityShell& EntityShell::operator=(const EntityShell& rhs)
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
    if( (b2Body*)NULL != body )
    {
        float xpos = body->GetPosition().x * sDescriptor::PIXELS_IN_METER;
        float ypos = body->GetPosition().y * sDescriptor::PIXELS_IN_METER;

        m_circle.setOutlineColor( GetColor() );

        if ( IsSelected() == true )
        {
            m_circle.setFillColor( sf::Color::Blue );
        }
        else
        {
            m_circle.setFillColor( sf::Color::Transparent );
        }

        // Drawing rays if configured
        if ( (bool)CastRays == true )
        {
            m_vision.Draw( window );
        }

        m_circle.setPosition( xpos, ypos);
        m_circle.setRotation(body->GetAngle() * 180.0F/b2_pi);

        m_triangle.setPosition( xpos, ypos);
        m_triangle.setRotation(body->GetAngle() * 180.0F/b2_pi);

        window.draw( m_circle );
        window.draw( m_triangle );
    }
}

/*****************************************************************************/
/**
  * @brief Return X coordinates in pixels
 **
******************************************************************************/
int EntityShell::GetX()
{
    if( GetDescriptor().Body == NULL ) return 0;

    return GetDescriptor().Body->GetPosition().x * sDescriptor::PIXELS_IN_METER;
}

/*****************************************************************************/
/**
  * @brief Return X coordinates in meters
 **
******************************************************************************/
float32 EntityShell::GetPhysicalX()
{
    if( GetDescriptor().Body == NULL ) return 0;

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
int EntityShell::GetY()
{
    if( GetDescriptor().Body == NULL ) return 0;

    return GetDescriptor().Body->GetPosition().y * sDescriptor::PIXELS_IN_METER;
}

/*****************************************************************************/
/**
  * @brief Return Y coordinates in meters
 **
******************************************************************************/
float32 EntityShell::GetPhysicalY()
{
    if( GetDescriptor().Body == NULL ) return 0;

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
float32 EntityShell::GetRotation()
{
    if( GetDescriptor().Body == NULL ) return 0;

    return GetDescriptor().Body->GetAngle() * (180.0/3.141592653589793238463);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const b2Vec2& EntityShell::GetPhysicalPoint()
{
    if( GetDescriptor().Body == NULL ) return m_zeroVector;

    return GetDescriptor().Body->GetPosition();
}
