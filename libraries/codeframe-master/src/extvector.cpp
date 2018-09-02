#include "extvector.hpp"

#include <MathUtilities.h>
#include <TextUtilities.h>

#include "base64.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
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

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<>
    std::string PropertyVector<float>::VectorToString( const std::vector<float>& vectorValue )
    {
        unsigned int vectorByteSize = vectorValue.size() * sizeof(float);
        const float* vectorValueData = &vectorValue[0];
        const uint8_t *indata = reinterpret_cast<const uint8_t *>(vectorValueData);
        std::string retValue = base64_encode(indata, vectorByteSize);
        return retValue;
    }

}
