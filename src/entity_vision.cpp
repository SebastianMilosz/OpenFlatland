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
    DrawRays         ( this, "DrawRays"         , false                 , cPropertyInfo().Kind(KIND_LOGIC).Description("DrawRays") ),
    RaysCnt          ( this, "RaysCnt"          , 100U                  , cPropertyInfo().Kind(KIND_NUMBER).Description("RaysCnt"), nullptr, std::bind(&EntityVision::SetRaysCnt, this, std::placeholders::_1) ),
    RaysSize         ( this, "RaysSize"         , 100U                  , cPropertyInfo().Kind(KIND_NUMBER).Description("RaysSize") ),
    RaysStartingAngle( this, "RaysStartingAngle", -45                   , cPropertyInfo().Kind(KIND_NUMBER).Description("RaysStartingAngle"), nullptr, std::bind(&EntityVision::SetRaysStartingAngle, this, std::placeholders::_1) ),
    RaysEndingAngle  ( this, "RaysEndingAngle"  ,  45                   , cPropertyInfo().Kind(KIND_NUMBER).Description("RaysEndingAngle"), nullptr, std::bind(&EntityVision::SetRaysEndingAngle, this, std::placeholders::_1) ),
    VisionVector     ( this, "VisionVector"     , thrust::host_vector<RayData>(), cPropertyInfo().Kind(KIND_VECTOR_THRUST_HOST, KIND_RAY_DATA).Description("VisionVector"), std::bind(&EntityVision::GetConstVisionVector, this), nullptr, std::bind(&EntityVision::GetVisionVector, this) )
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityVision::EntityVision( const EntityVision& other ) :
    Object( other ),
    DrawRays( other.DrawRays ),
    RaysCnt ( other.RaysCnt ),
    RaysSize( other.RaysSize ),
    RaysStartingAngle( other.RaysStartingAngle ),
    RaysEndingAngle( other.RaysEndingAngle ),
    VisionVector( other.VisionVector )
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
void EntityVision::CastRays(b2World& world, const b2Vec2& p1)
{
    StartFrame();

    unsigned int rayLength  ((unsigned int)RaysSize);
    unsigned int rayCntLimit((unsigned int)RaysCnt);
    float        rotation(getRotation().asRadians());
    volatile uint32_t fixture = 0U;

    int rayAngleStart((int)RaysStartingAngle);
    int rayAngleEnd((int)RaysEndingAngle);

    float currentRayAngle( TO_RADIAN( (std::min(rayAngleStart,rayAngleEnd)) ) ); //
    float rayAngleStep( TO_RADIAN((std::abs(std::max(rayAngleStart,rayAngleEnd) - std::min(rayAngleStart,rayAngleEnd))) / (float)rayCntLimit) );

    m_rayCastCallback.Reset();

    b2Vec2 p2;

    for ( unsigned int ray = 0U; ray < rayCntLimit; ray++ )
    {
        //calculate points of ray
        p2 = p1 + rayLength * b2Vec2( std::sin((-currentRayAngle-rotation)), std::cos((-currentRayAngle-rotation)) );

        world.RayCast( &m_rayCastCallback, p1, p2 );

        if ( m_rayCastCallback.WasHit() == true )
        {
            p2 = m_rayCastCallback.HitPoint();
            fixture = m_rayCastCallback.Fixture();
        }

        AddRay( EntityVision::Ray( p1, p2, fixture ) );
        currentRayAngle += rayAngleStep;
    }

#ifdef ENTITY_VISION_DEBUG
    p2 = p1 + b2Vec2( std::sin(-rotation), std::cos(-rotation) );
    AddDirectionRay( EntityVision::Ray( p1, p2, 0 ) );
#endif // ENTITY_VISION_DEBUG

    EndFrame();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::StartFrame()
{
    m_visionVector.clear();
    m_visionDataVector.clear();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::AddRay(EntityVision::Ray ray)
{
    m_visionVector.emplace_back( ray );
    m_visionDataVector.push_back( RayData( (ray.P2-ray.P1).Length(), ray.Fixture ) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setPosition(sf::Vector2f position)
{
    sf::Transformable::setPosition(position);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setRotation(sf::Angle angle)
{
    sf::Transformable::setRotation(angle);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
#ifdef ENTITY_VISION_DEBUG
void EntityVision::AddDirectionRay(EntityVision::Ray ray)
{
}
#endif // ENTITY_VISION_DEBUG

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::SetRaysCnt(const unsigned int cnt)
{
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
const thrust::host_vector<RayData>& EntityVision::GetConstVisionVector() const
{
    return m_visionDataVector;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
thrust::host_vector<RayData>& EntityVision::GetVisionVector()
{
    return m_visionDataVector;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::SetRaysStartingAngle(const int value)
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::SetRaysEndingAngle(const int value)
{
}
