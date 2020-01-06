#include "extvector.hpp"

#include <MathUtilities.h>
#include <TextUtilities.h>
#include <entity_vision_node.hpp>

#include "base64.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<>
    std::vector<EntityVisionNode::RayData> PropertyVector<EntityVisionNode::RayData>::VectorFromString( const std::string& value )
    {
        std::vector<EntityVisionNode::RayData> retVector;

        std::string outString = base64_decode( value );

        const char* byteTab = outString.c_str();

        const float* floatTab = reinterpret_cast<const float*>( byteTab );

        volatile unsigned int floatSize = outString.size() / sizeof( float );

        for ( unsigned int n = 0; n < floatSize; n++ )
        {
            retVector.push_back( EntityVisionNode::RayData(floatTab[ n ], 0U) );
        }

        return retVector;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<>
    std::string PropertyVector<EntityVisionNode::RayData>::VectorToString( const std::vector<EntityVisionNode::RayData>& vectorValue )
    {
        unsigned int vectorByteSize = vectorValue.size() * sizeof( float );
        const float* vectorValueData = &vectorValue[0].Distance;
        const uint8_t *indata = reinterpret_cast<const uint8_t *>(vectorValueData);
        std::string retValue = base64_encode(indata, vectorByteSize);
        return retValue;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<>
    std::vector<float> PropertyVector<float>::VectorFromString( const std::string& value )
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

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<>
    std::vector<unsigned int> PropertyVector<unsigned int>::VectorFromString( const std::string& value )
    {
        std::vector<unsigned int> retVector;

        std::string outString = base64_decode( value );

        const char* byteTab = outString.c_str();

        const unsigned int* floatTab = reinterpret_cast<const unsigned int*>( byteTab );

        volatile unsigned int floatSize = outString.size() / sizeof( unsigned int );

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
    std::string PropertyVector<unsigned int>::VectorToString( const std::vector<unsigned int>& vectorValue )
    {
        unsigned int vectorByteSize = vectorValue.size() * sizeof( unsigned int );
        const unsigned int* vectorValueData = &vectorValue[0];
        const uint8_t *indata = reinterpret_cast<const uint8_t *>(vectorValueData);
        std::string retValue = base64_encode(indata, vectorByteSize);
        return retValue;
    }
}
