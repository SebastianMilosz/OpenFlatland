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

        std::string outString = base64_decode( value );

        const char* byteTab = outString.c_str();

        const float* floatTab = reinterpret_cast<const float*>( byteTab );

        volatile unsigned int floatSize = outString.size() / sizeof( float );

        for ( unsigned int n = 0; n < floatSize; n++ )
        {
            retVector.push_back( floatTab[ n ] );
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
        unsigned int vectorByteSize = vectorValue.size() * sizeof( float );
        const float* vectorValueData = &vectorValue[0];
        const uint8_t *indata = reinterpret_cast<const uint8_t *>(vectorValueData);
        std::string retValue = base64_encode(indata, vectorByteSize);
        return retValue;
    }

}
