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
            static std::vector<T> VectorFromString( std::string value )
            {

                return std::vector<T>();
            }

            static std::string VectorToString( const std::vector<T>& point )
            {
                return "";
            }
    };
}

#endif // SERIALIZABLEPROPERTYTABLE_HPP_INCLUDED
