#ifndef TYPEINTERFACE_HPP_INCLUDED
#define TYPEINTERFACE_HPP_INCLUDED

#include <string>

namespace codeframe
{
    typedef int IntegerType;
    typedef double RealType;
    typedef std::string StringType;

    class TypeInterface
    {
        public:
            virtual void FromStringCallback ( const StringType&  value ) = 0;
            virtual void FromIntegerCallback( const IntegerType& value ) = 0;
            virtual void FromRealCallback   ( const RealType&    value ) = 0;

            virtual StringType  ToStringCallback () const = 0;
            virtual IntegerType ToIntegerCallback() const = 0;
            virtual RealType    ToRealCallback   () const = 0;
    };
}

#endif // TYPEINTERFACE_HPP_INCLUDED
