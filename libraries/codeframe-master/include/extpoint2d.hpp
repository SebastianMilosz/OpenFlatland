#ifndef EXTENDEDTYPE2DPOINT_HPP_INCLUDED
#define EXTENDEDTYPE2DPOINT_HPP_INCLUDED

#include "typeinterface.hpp"

namespace codeframe
{
    class Point2D : public TypeInterface
    {
        public:
                     Point2D();
                     Point2D( int x, int y );
                     Point2D( const Point2D& other );
            virtual ~Point2D();

            virtual void FromStringCallback ( StringType  value );
            virtual void FromIntegerCallback( IntegerType value );
            virtual void FromRealCallback( RealType value );

            virtual StringType ToStringCallback() const;
            virtual IntegerType ToIntegerCallback() const;
            virtual RealType ToRealCallback() const;

            virtual Point2D& operator=(const Point2D& other);
            virtual Point2D& operator+(const Point2D& rhs);
            virtual bool     operator==(const Point2D& sval);
            virtual bool     operator!=(const Point2D& sval);

            int X() { return m_x; }
            int Y() { return m_y; }

        private:
            int m_x;
            int m_y;
    };

    typedef Point2D Point2D;
}

#endif // EXTENDEDTYPE2DPOINT_HPP_INCLUDED