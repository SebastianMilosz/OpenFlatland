#ifndef EXTENDEDTYPE2DPOINT_HPP_INCLUDED
#define EXTENDEDTYPE2DPOINT_HPP_INCLUDED

#include "typeinterface.hpp"

#include <SFML/Graphics.hpp>

namespace codeframe
{
    template<typename T>
    class Point2D : public TypeInterface
    {
        public:
            Point2D() : m_x( 0 ), m_y( 0 ) {}
            Point2D( T x, T y ) : m_x( x ), m_y( y ) {}
            Point2D( sf::Vector2f point ) : m_x( point.x ), m_y( point.y ) {}
            Point2D( const Point2D& other ) : m_x( other.m_x ), m_y( other.m_y ) {}
            virtual ~Point2D() {}

            virtual void FromStringCallback ( const StringType&  value );
            virtual void FromIntegerCallback( const IntegerType& value );
            virtual void FromRealCallback   ( const RealType& value );

            virtual StringType ToStringCallback() const;
            virtual IntegerType ToIntegerCallback() const;
            virtual RealType ToRealCallback() const;

            virtual Point2D& operator=(const Point2D& other);
            virtual Point2D& operator+(const Point2D& rhs);
            virtual bool     operator==(const Point2D& sval) const;
            virtual bool     operator!=(const Point2D& sval) const;

            T X() const { return m_x; }
            T Y() const { return m_y; }

        static Point2D<T> Point2DFromString( const std::string& value )
        {
            Point2D<T> retType(0,0);
            retType.FromStringCallback( value );
            return retType;
        }

        static std::string Point2DToString( const Point2D<T>& point )
        {
            return point.ToStringCallback();
        }

        static IntegerType Point2DToInt( const Point2D<T>& value )
        {
            return 0;
        }

        static Point2D<T> Point2AddOperator( const Point2D<T>& value1, const Point2D<T>& value2 )
        {
            return value1;
        }

        private:
            T m_x;
            T m_y;
    };
}

#endif // EXTENDEDTYPE2DPOINT_HPP_INCLUDED
