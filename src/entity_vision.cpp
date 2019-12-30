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
    RaysCnt          ( this, "RaysCnt"          , 100U                , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysCnt"), nullptr, std::bind(&EntityVision::SetRaysCnt, this, std::placeholders::_1) ),
    RaysSize         ( this, "RaysSize"         , 100U                , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysSize") ),
    RaysStartingAngle( this, "RaysStartingAngle", -45                 , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysStartingAngle"), nullptr, std::bind(&EntityVision::SetRaysStartingAngle, this, std::placeholders::_1) ),
    RaysEndingAngle  ( this, "RaysEndingAngle"  ,  45                 , cPropertyInfo().Kind( KIND_NUMBER ).Description("RaysEndingAngle"), nullptr, std::bind(&EntityVision::SetRaysEndingAngle, this, std::placeholders::_1) ),
    VisionVector     ( this, "VisionVector"     , std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).ReferencePath("../ANN/AnnLayer[0].Input").Description("VisionVector"), std::bind(&EntityVision::GetDistanceVector, this) ),
    FixtureVector    ( this, "FixtureVector"    , std::vector<float>(), cPropertyInfo().Kind( KIND_VECTOR ).ReferencePath("../ANN/AnnLayer[1].Input").Description("FixtureVector"), std::bind(&EntityVision::GetFixtureVector, this) )
{
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
void EntityVision::AddRay(EntityVision::sRay ray)
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
void EntityVision::setPosition(const float x, const float y)
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::setRotation(const float angle)
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void EntityVision::AddDirectionRay(EntityVision::sRay ray)
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
