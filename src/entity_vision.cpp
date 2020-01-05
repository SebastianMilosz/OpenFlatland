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
    DrawRays         ( this, "DrawRays"         , false                 , cPropertyInfo().Kind( KIND_LOGIC  ).Description("DrawRays") ),
    RaysCnt          ( this, "RaysCnt"          , 100U                  , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysCnt"), nullptr, std::bind(&EntityVision::SetRaysCnt, this, std::placeholders::_1) ),
    RaysSize         ( this, "RaysSize"         , 100U                  , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysSize") ),
    RaysStartingAngle( this, "RaysStartingAngle", -45                   , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysStartingAngle"), nullptr, std::bind(&EntityVision::SetRaysStartingAngle, this, std::placeholders::_1) ),
    RaysEndingAngle  ( this, "RaysEndingAngle"  ,  45                   , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysEndingAngle"), nullptr, std::bind(&EntityVision::SetRaysEndingAngle, this, std::placeholders::_1) ),
    VisionVector     ( this, "VisionVector"     , std::vector<RayData>(), cPropertyInfo().Kind( KIND_VECTOR ).ReferencePath("../ANN/AnnLayer[0].Input").Description("VisionVector"), std::bind(&EntityVision::GetVisionVector, this) )
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

    register unsigned int rayLength  ( (unsigned int)RaysSize );
    register unsigned int rayCntLimit( (unsigned int)RaysCnt  );
    register float32      rotation( TO_RADIAN( getRotation() ) );

    int rayAngleStart( (int)RaysStartingAngle );
    int rayAngleEnd( (int)RaysEndingAngle );

    register float32 currentRayAngle( TO_RADIAN( (std::min(rayAngleStart,rayAngleEnd)) ) ); //
    register float32 rayAngleStep( TO_RADIAN((std::abs(std::max(rayAngleStart,rayAngleEnd) - std::min(rayAngleStart,rayAngleEnd))) / (float32)rayCntLimit) );

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
        }

        AddRay( EntityVision::Ray( p1, p2, 0 ) );
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
    m_visionDataVector.emplace_back( (ray.P2-ray.P1).Length(), ray.Fixture );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setPosition(float x, float y)
{
    sf::Transformable::setPosition(x, y);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setRotation(float angle)
{
    sf::Transformable::setRotation(angle);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::AddDirectionRay(EntityVision::Ray ray)
{
}

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
const std::vector<EntityVision::RayData>& EntityVision::GetVisionVector() const
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
