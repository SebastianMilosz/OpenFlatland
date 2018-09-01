#include "extvector.hpp"

#include <MathUtilities.h>
#include <TextUtilities.h>

namespace codeframe
{
    template<>
    std::vector<float> PropertyVector<float>::VectorFromString( std::string value )
    {
        std::vector<float> retVector;

        // Split using ; separator
        std::vector<std::string> vectorPartsStrings;
        utilities::text::split(value, ";", vectorPartsStrings);

        if ( vectorPartsStrings.size() > 0 )
        {
            for ( std::vector<std::string>::iterator it = vectorPartsStrings.begin(); it != vectorPartsStrings.end(); ++it )
            {

            }
        }
        return retVector;
    }

    template<>
    std::string PropertyVector<float>::VectorToString( const std::vector<float>& point )
    {
        return "";
    }

}
