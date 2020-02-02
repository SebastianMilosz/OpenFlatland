#ifndef RAY_DATA_HPP
#define RAY_DATA_HPP

#include <cstdint>
#include <cmath>

struct RayData
{
    RayData();
    RayData(const float distance, const uint32_t fixture);

    bool operator==(const RayData& rval) const
    {
        if (  std::fabs(Distance - rval.Distance) > 0.001F && Fixture == rval.Fixture )
        {
            return true;
        }
        return false;
    }

    float    Distance;
    uint32_t Fixture;
};

#endif // RAY_DATA_HPP
