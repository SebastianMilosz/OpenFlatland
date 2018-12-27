#ifndef EXTTHRUST_HPP_INCLUDED
#define EXTTHRUST_HPP_INCLUDED

#include "typeinterface.hpp"

#include <thrust/device_vector.h>
#include <string>

namespace codeframe
{
    template<typename T>
    class PropertyThrustVector
    {
        public:
            static thrust::host_vector<T> VectorFromString( const std::string& value );
            static std::string VectorToString( const thrust::host_vector<T>& vectorValue );
    };
}

#endif // EXTTHRUST_HPP_INCLUDED
