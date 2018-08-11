#ifndef EXTENDEDTYPE2DPOINT_HPP_INCLUDED
#define EXTENDEDTYPE2DPOINT_HPP_INCLUDED

#include "extendedtypeinterface.hpp"

namespace codeframe
{

    class Point2D : public ExtTypeInterface
    {
        public:
            Point2D()
            {

            }

            virtual ~Point2D()
            {

            }

            virtual void FromStringCallback ( StringType  value )
            {

            }

            virtual void FromIntegerCallback( IntegerType value )
            {

            }

            virtual void FromRealCallback( RealType value )
            {

            }

            virtual StringType ToStringCallback() const
            {
                return std::string("NONE");
            }

            virtual IntegerType ToIntegerCallback() const
            {
                return 0;
            }

            virtual RealType ToRealCallback() const
            {
                return 0.0F;
            }

            virtual Point2D&  operator+(const Point2D& rhs)
            {
                return *this;
            }

            virtual bool operator==(const Point2D& sval)
            {
                return false;
            }

            virtual bool operator!=(const Point2D& sval)
            {
                return false;
            }
    };

}

#endif // EXTENDEDTYPE2DPOINT_HPP_INCLUDED
