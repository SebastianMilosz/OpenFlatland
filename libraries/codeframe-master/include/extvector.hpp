#ifndef SERIALIZABLEPROPERTYTABLE_HPP_INCLUDED
#define SERIALIZABLEPROPERTYTABLE_HPP_INCLUDED

#include "typeinterface.hpp"

#include <vector>
#include <string>

namespace codeframe
{
    template<typename T>
    class PropertyVector
    {
        public:
            static std::vector<T> VectorFromString( const std::string& value );
            static std::string VectorToString( const std::vector<T>& vectorValue );
    };
}

#endif // SERIALIZABLEPROPERTYTABLE_HPP_INCLUDED
