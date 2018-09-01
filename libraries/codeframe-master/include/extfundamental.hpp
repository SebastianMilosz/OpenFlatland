#ifndef EXTFUNDAMENTAL_HPP_INCLUDED
#define EXTFUNDAMENTAL_HPP_INCLUDED

#include "typeinterface.hpp"

#include <string>

namespace codeframe
{
    template<typename T>
    class FundamentalTypes
    {
        public:
            static bool BoolFromString( std::string value )
            {
                int retVal = utilities::math::StrToInt( value );
                return (bool)retVal;
            }

            static int IntFromString( std::string value )
            {
                int retVal = utilities::math::StrToInt( value );
                return retVal;
            }

            static unsigned int UIntFromString( std::string value )
            {
                unsigned int retVal = utilities::math::StrToInt( value );
                return retVal;
            }

            static float FloatFromString( std::string value )
            {
                unsigned int retVal = utilities::math::StrToFloat( value );
                return retVal;
            }

            static double DoubleFromString( std::string value )
            {
                unsigned int retVal = utilities::math::StrToDouble( value );
                return retVal;
            }

            static std::string StringFromString( std::string value )
            {
                return value;
            }

            static std::string BoolToString( const bool& value )
            {
                return utilities::math::IntToStr( value );
            }

            static std::string IntToString( const int& value )
            {
                return utilities::math::IntToStr( value );
            }

            static std::string UIntToString( const unsigned int& value )
            {
                return utilities::math::IntToStr( value );
            }

            static std::string FloatToString( const float& value )
            {
                return utilities::math::FloatToStr( value );
            }

            static std::string DoubleToString( const double& value )
            {
                return utilities::math::DoubleToStr( value );
            }

            static std::string StringToString( const std::string& value )
            {
                return value;
            }

            static IntegerType BoolToInt( const bool& value )
            {
                return value;
            }

            static IntegerType IntToInt( const int& value )
            {
                return value;
            }

            static IntegerType UIntToInt( const unsigned int& value )
            {
                return value;
            }

            static IntegerType FloatToInt( const float& value )
            {
                return value;
            }

            static IntegerType DoubleToInt( const double& value )
            {
                return value;
            }

            static IntegerType StringToInt( const std::string& value )
            {
                return utilities::math::StrToInt( value );
            }

            static bool BoolFromInt( IntegerType value )
            {
                return value;
            }

            static int IntFromInt( IntegerType value )
            {
                return value;
            }

            static unsigned int UIntFromInt( IntegerType value )
            {
                unsigned int retVal = value;
                return retVal;
            }

            static float FloatFromInt( IntegerType value )
            {
                return value;
            }

            static double DoubleFromInt( IntegerType value )
            {
                return value;
            }

            static std::string StringFromInt( IntegerType value )
            {
                std::string retVal = utilities::math::IntToStr( value );
                return retVal;
            }

            static RealType BoolToReal( const bool& value )
            {
                return value;
            }

            static RealType IntToReal( const int& value )
            {
                return value;
            }

            static RealType UIntToReal( const unsigned int& value )
            {
                return value;
            }

            static RealType FloatToReal( const float& value )
            {
                return value;
            }

            static RealType DoubleToReal( const double& value )
            {
                return value;
            }

            static RealType StringToReal( const std::string& value )
            {
                return 0;
            }

            static bool BoolFromReal( double value )
            {
                int retVal = value;
                return retVal;
            }

            static int IntFromReal( double value )
            {
                int retVal = value;
                return retVal;
            }

            static unsigned int UIntFromReal( double value )
            {
                unsigned int retVal = value;
                return retVal;
            }

            static float FloatFromReal( double value )
            {
                return value;
            }

            static double DoubleFromReal( double value )
            {
                return value;
            }

            static std::string StringFromReal( double value )
            {
                std::string retVal = std::string("0");
                return retVal;
            }

            static bool BoolAddOperator( const bool& value1, const bool& value2 )
            {
                return (value1 + value2);
            }

            static int IntAddOperator( const int& value1, const int& value2 )
            {
                return (value1 + value2);
            }

            static unsigned int UIntAddOperator( const unsigned int& value1, const unsigned int& value2 )
            {
                return (value1 + value2);
            }

            static float FloatAddOperator( const float& value1, const float& value2 )
            {
                return (value1 + value2);
            }

            static double DoubleAddOperator( const double& value1, const double& value2 )
            {
                return (value1 + value2);
            }

            static std::string StringAddOperator( const std::string& value1, const std::string& value2 )
            {
                return (value1 + value2);
            }
    };
}

#endif // EXTFUNDAMENTAL_HPP_INCLUDED
