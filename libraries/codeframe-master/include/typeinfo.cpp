#include "typeinfo.hpp"

#include <MathUtilities.h>

namespace codeframe
{
    int IntFromString( std::string value )
    {
        return utilities::math::StrToInt( value );
    }

    unsigned int UIntFromString( std::string value )
    {
        return utilities::math::StrToInt( value );
    }

    std::string StringFromString( std::string value )
    {
        return value;
    }

    std::string IntToString( int value )
    {
        return utilities::math::IntToStr( value );
    }

    std::string UIntToString( unsigned int value )
    {
        return utilities::math::IntToStr( value );
    }

    std::string StringToString( std::string value )
    {
        return value;
    }

    IntegerType IntToInt( int value )
    {
        return value;
    }

    IntegerType UIntToInt( unsigned int value )
    {
        return value;
    }

    IntegerType StringToInt( std::string value )
    {
        return utilities::math::StrToInt( value );
    }

    int IntFromInt( IntegerType value )
    {
        return value;
    }

    unsigned int UIntFromInt( IntegerType value )
    {
        return value;
    }

    std::string StringFromInt( IntegerType value )
    {
        return utilities::math::IntToStr( value );
    }

    double IntToReal( int value )
    {
        return value;
    }

    double UIntToReal( unsigned int value )
    {
        return value;
    }

    double StringToReal( std::string value )
    {
        return 0;
    }

    int IntFromReal( double value )
    {
        return value;
    }

    unsigned int UIntFromReal( double value )
    {
        return value;
    }

    std::string StringFromReal( double value )
    {
        return "0";
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
    void TypeInfo<T>::SetToStringCallback( StringType (*toStringCallback)( T value ) )
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
    void TypeInfo<T>::SetToIntegerCallback( IntegerType (*toIntegerCallback)( T value ) )
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
    void TypeInfo<T>::SetToRealCallback( RealType (*toRealCallback)( T value ) )
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
    REGISTER_TYPE( std::string    , "text" );
    REGISTER_TYPE( int            , "int"  );
    REGISTER_TYPE( unsigned int   , "int"  );
    REGISTER_TYPE( short          , "int"  );
    REGISTER_TYPE( unsigned short , "int"  );
    REGISTER_TYPE( float          , "real" );
    REGISTER_TYPE( double         , "real" );

    void CODEFRAME_TYPES_INITIALIZE( void )
    {
        GetTypeInfo<int         >().SetFromStringCallback( &IntFromString    );
        GetTypeInfo<unsigned int>().SetFromStringCallback( &UIntFromString   );
        GetTypeInfo<std::string >().SetFromStringCallback( &StringFromString );

        GetTypeInfo<int         >().FromString( "" );
        GetTypeInfo<unsigned int>().FromString( "" );
        GetTypeInfo<std::string >().FromString( "" );

        GetTypeInfo<int         >().SetToStringCallback( &IntToString    );
        GetTypeInfo<unsigned int>().SetToStringCallback( &UIntToString   );
        GetTypeInfo<std::string >().SetToStringCallback( &StringToString );

        GetTypeInfo<int         >().ToString( 0  );
        GetTypeInfo<unsigned int>().ToString( 0  );
        GetTypeInfo<std::string >().ToString( "" );

        GetTypeInfo<int         >().SetFromIntegerCallback( &IntFromInt    );
        GetTypeInfo<unsigned int>().SetFromIntegerCallback( &UIntFromInt   );
        GetTypeInfo<std::string >().SetFromIntegerCallback( &StringFromInt );

        GetTypeInfo<int         >().FromInteger( 0 );
        GetTypeInfo<unsigned int>().FromInteger( 0 );
        GetTypeInfo<std::string >().FromInteger( 0 );

        GetTypeInfo<int         >().SetToIntegerCallback( &IntToInt    );
        GetTypeInfo<unsigned int>().SetToIntegerCallback( &UIntToInt   );
        GetTypeInfo<std::string >().SetToIntegerCallback( &StringToInt );

        GetTypeInfo<int         >().ToInteger( 0  );
        GetTypeInfo<unsigned int>().ToInteger( 0  );
        GetTypeInfo<std::string >().ToInteger( "" );

        GetTypeInfo<int         >().SetFromRealCallback( &IntFromReal    );
        GetTypeInfo<unsigned int>().SetFromRealCallback( &UIntFromReal   );
        GetTypeInfo<std::string >().SetFromRealCallback( &StringFromReal );

        GetTypeInfo<int         >().FromReal( 0 );
        GetTypeInfo<unsigned int>().FromReal( 0 );
        GetTypeInfo<std::string >().FromReal( 0 );

        GetTypeInfo<int         >().SetToRealCallback( &IntToReal    );
        GetTypeInfo<unsigned int>().SetToRealCallback( &UIntToReal   );
        GetTypeInfo<std::string >().SetToRealCallback( &StringToReal );

        GetTypeInfo<int         >().ToReal( 0 );
        GetTypeInfo<unsigned int>().ToReal( 0 );
        GetTypeInfo<std::string >().ToReal( "" );
    }
}
