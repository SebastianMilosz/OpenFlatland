#ifndef RAY_DATA_HPP
#define RAY_DATA_HPP

#include <MathUtilities.h>
#include <cstdint>
#include <cmath>

struct RayData
{
    RayData();
    RayData(const float distance, const uint32_t fixture = 0U);

    bool operator==(const RayData& rval) const
    {
        if (  std::fabs(Distance - rval.Distance) > 0.001F && Fixture == rval.Fixture )
        {
            return true;
        }
        return false;
    }

    // Comparison operators
    bool operator!=(const RayData& rval) const
    {
        if (  std::fabs(Distance - rval.Distance) > 0.001F && Fixture == rval.Fixture )
        {
            return false;
        }
        return true;
    }

    // Copy constructor
    RayData( const RayData& sval ) :
        Distance(sval.Distance),
        Fixture(sval.Fixture)
    {
    }

    // Assignment  operator
    RayData& operator=( const RayData& val )
    {
        Distance = val.Distance;
        Fixture = val.Fixture;
        return *this;
    }

    operator std::string() const
    {
        return utilities::math::FloatToStr(Distance);
    }

    float    Distance;
    uint32_t Fixture;
};

#endif // RAY_DATA_HPP
