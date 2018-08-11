#include "typeinfo.hpp"

// Extended types
#include "extendedtype2dpoint.hpp"

#include <MathUtilities.h>

namespace codeframe
{
    int IntFromString( std::string value )
    {
        int retVal = utilities::math::StrToInt( value );
        return retVal;
    }

    unsigned int UIntFromString( std::string value )
    {
        unsigned int retVal = utilities::math::StrToInt( value );
        return retVal;
    }

    std::string StringFromString( std::string value )
    {
        return value;
    }

    Point2D Point2DFromString( std::string value )
    {
        Point2D retType;
        retType.FromStringCallback( value );
        return retType;
    }

    std::string IntToString( const int& value )
    {
        return utilities::math::IntToStr( value );
    }

    std::string UIntToString( const unsigned int& value )
    {
        return utilities::math::IntToStr( value );
    }

    std::string StringToString( const std::string& value )
    {
        return value;
    }

    std::string Point2DToString( const Point2D& point )
    {
        return point.ToStringCallback();
    }

    IntegerType IntToInt( const int& value )
    {
        return value;
    }

    IntegerType UIntToInt( const unsigned int& value )
    {
        return value;
    }

    IntegerType StringToInt( const std::string& value )
    {
        return utilities::math::StrToInt( value );
    }

    int IntFromInt( IntegerType value )
    {
        return value;
    }

    unsigned int UIntFromInt( IntegerType value )
    {
        unsigned int retVal = value;
        return retVal;
    }

    std::string StringFromInt( IntegerType value )
    {
        std::string retVal = utilities::math::IntToStr( value );
        return retVal;
    }

    RealType IntToReal( const int& value )
    {
        return value;
    }

    RealType UIntToReal( const unsigned int& value )
    {
        return value;
    }

    RealType StringToReal( const std::string& value )
    {
        return 0;
    }

    int IntFromReal( double value )
    {
        int retVal = value;
        return retVal;
    }

    unsigned int UIntFromReal( double value )
    {
        unsigned int retVal = value;
        return retVal;
    }

    std::string StringFromReal( double value )
    {
        std::string retVal = std::string("0");
        return retVal;
    }

    template<typename T>
    TypeInfo<T>::TypeInfo( const char* typeName, const char* typeUser, const eType enumType ) :
        TypeCompName( typeName ),
        TypeUserName( typeUser ),
        TypeCode( enumType ),
        FromStringCallback( NULL ),
        ToStringCallback( NULL ),
        FromIntegerCallback( NULL ),
        ToIntegerCallback( NULL ),
        FromRealCallback( NULL ),
        ToRealCallback( NULL )
    {
    }

    template<typename T>
    const eType TypeInfo<T>::StringToTypeCode( std::string typeText )
    {
        if( typeText == "int"  ) return TYPE_INT;
        if( typeText == "real" ) return TYPE_REAL;
        if( typeText == "text" ) return TYPE_TEXT;
        if( typeText == "ext"  ) return TYPE_EXTENDED;

        return TYPE_NON;
    }

    template<typename T>
    void TypeInfo<T>::SetFromStringCallback( T (*fromStringCallback)( StringType value ) )
    {
        FromStringCallback = fromStringCallback;
    }

    template<typename T>
    T TypeInfo<T>::FromString( StringType value )
    {
        if ( NULL != FromStringCallback )
        {
            return FromStringCallback( value );
        }
        return T();
    }

    template<typename T>
    void TypeInfo<T>::SetToStringCallback( StringType (*toStringCallback)( const T& value ) )
    {
        ToStringCallback = toStringCallback;
    }

    template<typename T>
    StringType TypeInfo<T>::ToString( T value )
    {
        if ( NULL != ToStringCallback )
        {
            return ToStringCallback( value );
        }
        return StringType("");
    }

    template<typename T>
    void TypeInfo<T>::SetFromIntegerCallback( T (*fromIntegerCallback)( IntegerType value ) )
    {
        FromIntegerCallback = fromIntegerCallback;
    }

    template<typename T>
    T TypeInfo<T>::FromInteger( IntegerType value )
    {
        if ( NULL != FromIntegerCallback )
        {
            return FromIntegerCallback( value );
        }
        return T();
    }

    template<typename T>
    void TypeInfo<T>::SetToIntegerCallback( IntegerType (*toIntegerCallback)( const T& value ) )
    {
        ToIntegerCallback = toIntegerCallback;
    }

    template<typename T>
    IntegerType TypeInfo<T>::ToInteger( T value )
    {
        if ( NULL != ToIntegerCallback )
        {
            return ToIntegerCallback( value );
        }
        return IntegerType();
    }

    template<typename T>
    void TypeInfo<T>::SetFromRealCallback( T (*fromRealCallback)( RealType value ) )
    {
        FromRealCallback = fromRealCallback;
    }

    template<typename T>
    T TypeInfo<T>::FromReal( RealType value )
    {
        if ( NULL != FromRealCallback )
        {
            return FromRealCallback( value );
        }
        return T();
    }

    template<typename T>
    void TypeInfo<T>::SetToRealCallback( RealType (*toRealCallback)( const T& value ) )
    {
        ToRealCallback = toRealCallback;
    }

    template<typename T>
    RealType TypeInfo<T>::ToReal( T value )
    {
        if ( NULL != ToRealCallback )
        {
            return ToRealCallback( value );
        }
        return 0.0F;
    }

    // Fundamental types
    REGISTER_TYPE( std::string   , "text"    );
    REGISTER_TYPE( int           , "int"     );
    REGISTER_TYPE( unsigned int  , "int"     );
    REGISTER_TYPE( short         , "int"     );
    REGISTER_TYPE( unsigned short, "int"     );
    REGISTER_TYPE( float         , "real"    );
    REGISTER_TYPE( double        , "real"    );
    REGISTER_TYPE( Point2D       , "point2d" );

    TypeInitializer::TypeInitializer( void )
    {
        GetTypeInfo<int         >().SetFromStringCallback( &IntFromString     );
        GetTypeInfo<unsigned int>().SetFromStringCallback( &UIntFromString    );
        GetTypeInfo<std::string >().SetFromStringCallback( &StringFromString  );
        GetTypeInfo<Point2D     >().SetFromStringCallback( &Point2DFromString );

        GetTypeInfo<int         >().FromString( "" );
        GetTypeInfo<unsigned int>().FromString( "" );
        GetTypeInfo<std::string >().FromString( "" );
        GetTypeInfo<Point2D     >().FromString( "" );

        GetTypeInfo<int         >().SetToStringCallback( &IntToString     );
        GetTypeInfo<unsigned int>().SetToStringCallback( &UIntToString    );
        GetTypeInfo<std::string >().SetToStringCallback( &StringToString  );
        GetTypeInfo<Point2D     >().SetToStringCallback( &Point2DToString );

        GetTypeInfo<int         >().ToString( 0  );
        GetTypeInfo<unsigned int>().ToString( 0  );
        GetTypeInfo<std::string >().ToString( "" );
        GetTypeInfo<Point2D     >().ToString( Point2D() );

        GetTypeInfo<int         >().SetFromIntegerCallback( &IntFromInt    );
        GetTypeInfo<unsigned int>().SetFromIntegerCallback( &UIntFromInt   );
        GetTypeInfo<std::string >().SetFromIntegerCallback( &StringFromInt );

        GetTypeInfo<int         >().FromInteger( 0 );
        GetTypeInfo<unsigned int>().FromInteger( 0 );
        GetTypeInfo<std::string >().FromInteger( 0 );
        GetTypeInfo<Point2D     >().FromInteger( 0 );

        GetTypeInfo<int         >().SetToIntegerCallback( &IntToInt    );
        GetTypeInfo<unsigned int>().SetToIntegerCallback( &UIntToInt   );
        GetTypeInfo<std::string >().SetToIntegerCallback( &StringToInt );

        GetTypeInfo<int         >().ToInteger( 0  );
        GetTypeInfo<unsigned int>().ToInteger( 0  );
        GetTypeInfo<std::string >().ToInteger( "" );
        GetTypeInfo<Point2D     >().ToInteger( Point2D() );

        GetTypeInfo<int         >().SetFromRealCallback( &IntFromReal    );
        GetTypeInfo<unsigned int>().SetFromRealCallback( &UIntFromReal   );
        GetTypeInfo<std::string >().SetFromRealCallback( &StringFromReal );

        GetTypeInfo<int         >().FromReal( 0 );
        GetTypeInfo<unsigned int>().FromReal( 0 );
        GetTypeInfo<std::string >().FromReal( 0 );
        GetTypeInfo<Point2D     >().FromReal( 0 );

        GetTypeInfo<int         >().SetToRealCallback( &IntToReal    );
        GetTypeInfo<unsigned int>().SetToRealCallback( &UIntToReal   );
        GetTypeInfo<std::string >().SetToRealCallback( &StringToReal );

        GetTypeInfo<int         >().ToReal( 0 );
        GetTypeInfo<unsigned int>().ToReal( 0 );
        GetTypeInfo<std::string >().ToReal( "" );
        GetTypeInfo<Point2D     >().ToReal( Point2D() );
    }
}
