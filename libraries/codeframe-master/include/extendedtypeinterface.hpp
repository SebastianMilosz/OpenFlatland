#ifndef EXTENDEDTYPEINTERFACE_HPP_INCLUDED
#define EXTENDEDTYPEINTERFACE_HPP_INCLUDED

#include "typeinfo.hpp"

namespace codeframe
{

    class ExtTypeInterface
    {
        public:
            virtual void FromStringCallback ( StringType  value ) = 0;
            virtual void FromIntegerCallback( IntegerType value ) = 0;
            virtual void FromRealCallback   ( RealType    value ) = 0;

            virtual StringType  ToStringCallback () const = 0;
            virtual IntegerType ToIntegerCallback() const = 0;
            virtual RealType    ToRealCallback   () const = 0;
    };

}

#endif // EXTENDEDTYPEINTERFACE_HPP_INCLUDED
