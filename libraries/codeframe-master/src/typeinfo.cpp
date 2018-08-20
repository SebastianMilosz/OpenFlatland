#include "typeinfo.hpp"

#include <MathUtilities.h>

namespace codeframe
{
    bool BoolFromString( std::string value )
    {
        int retVal = utilities::math::StrToInt( value );
        return (bool)retVal;
    }

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
        Point2D retType(0,0);
        retType.FromStringCallback( value );
        return retType;
    }

    std::string BoolToString( const bool& value )
    {
        return utilities::math::IntToStr( value );
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

    IntegerType BoolToInt( const bool& value )
    {
        return value;
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

    IntegerType Point2DToInt( const Point2D& value )
    {
        return 0;
    }

    bool BoolFromInt( IntegerType value )
    {
        return value;
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

    RealType BoolToReal( const bool& value )
    {
        return value;
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

    bool BoolFromReal( double value )
    {
        int retVal = value;
        return retVal;
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
        if( typeText == "ivec" ) return TYPE_IVECTOR;

        return TYPE_NON;
    }

    template<typename T>
    void TypeInfo<T>::SetFromStringCallback( T (*fromStringCallback)( StringType value ) )
    {
        FromStringCallback = fromStringCallback;
    }

    template<typename T>
    void TypeInfo<T>::SetToStringCallback( StringType (*toStringCallback)( const T& value ) )
    {
        ToStringCallback = toStringCallback;
    }

    template<typename T>
    void TypeInfo<T>::SetFromIntegerCallback( T (*fromIntegerCallback)( IntegerType value ) )
    {
        FromIntegerCallback = fromIntegerCallback;
    }

    template<typename T>
    void TypeInfo<T>::SetToIntegerCallback( IntegerType (*toIntegerCallback)( const T& value ) )
    {
        ToIntegerCallback = toIntegerCallback;
    }

    template<typename T>
    void TypeInfo<T>::SetFromRealCallback( T (*fromRealCallback)( RealType value ) )
    {
        FromRealCallback = fromRealCallback;
    }

    template<typename T>
    void TypeInfo<T>::SetToRealCallback( RealType (*toRealCallback)( const T& value ) )
    {
        ToRealCallback = toRealCallback;
    }

    // Fundamental types
    REGISTER_TYPE( std::string   , "text" );
    REGISTER_TYPE( bool          , "int"  );
    REGISTER_TYPE( int           , "int"  );
    REGISTER_TYPE( unsigned int  , "int"  );
    REGISTER_TYPE( short         , "int"  );
    REGISTER_TYPE( unsigned short, "int"  );
    REGISTER_TYPE( float         , "real" );
    REGISTER_TYPE( double        , "real" );
    REGISTER_TYPE( Point2D       , "ivec" );

    TypeInitializer::TypeInitializer( void )
    {
        GetTypeInfo<bool        >().SetFromStringCallback( &BoolFromString    );
        GetTypeInfo<int         >().SetFromStringCallback( &IntFromString     );
        GetTypeInfo<unsigned int>().SetFromStringCallback( &UIntFromString    );
        GetTypeInfo<std::string >().SetFromStringCallback( &StringFromString  );
        GetTypeInfo<Point2D     >().SetFromStringCallback( &Point2DFromString );

        GetTypeInfo<bool        >().SetToStringCallback( &BoolToString    );
        GetTypeInfo<int         >().SetToStringCallback( &IntToString     );
        GetTypeInfo<unsigned int>().SetToStringCallback( &UIntToString    );
        GetTypeInfo<std::string >().SetToStringCallback( &StringToString  );
        GetTypeInfo<Point2D     >().SetToStringCallback( &Point2DToString );

        GetTypeInfo<bool        >().SetFromIntegerCallback( &BoolFromInt   );
        GetTypeInfo<int         >().SetFromIntegerCallback( &IntFromInt    );
        GetTypeInfo<unsigned int>().SetFromIntegerCallback( &UIntFromInt   );
        GetTypeInfo<std::string >().SetFromIntegerCallback( &StringFromInt );
        GetTypeInfo<Point2D     >().SetFromIntegerCallback( NULL );

        GetTypeInfo<bool        >().SetToIntegerCallback( &BoolToInt    );
        GetTypeInfo<int         >().SetToIntegerCallback( &IntToInt     );
        GetTypeInfo<unsigned int>().SetToIntegerCallback( &UIntToInt    );
        GetTypeInfo<std::string >().SetToIntegerCallback( &StringToInt  );
        GetTypeInfo<Point2D     >().SetToIntegerCallback( &Point2DToInt );

        GetTypeInfo<bool        >().SetFromRealCallback( &BoolFromReal   );
        GetTypeInfo<int         >().SetFromRealCallback( &IntFromReal    );
        GetTypeInfo<unsigned int>().SetFromRealCallback( &UIntFromReal   );
        GetTypeInfo<std::string >().SetFromRealCallback( &StringFromReal );

        GetTypeInfo<bool        >().SetToRealCallback( &BoolToReal   );
        GetTypeInfo<int         >().SetToRealCallback( &IntToReal    );
        GetTypeInfo<unsigned int>().SetToRealCallback( &UIntToReal   );
        GetTypeInfo<std::string >().SetToRealCallback( &StringToReal );
    }
}
