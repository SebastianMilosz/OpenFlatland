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
        return "NONE";
    }

    std::string UIntToString( unsigned int value )
    {
        return "NONE";
    }

    std::string StringToString( std::string value )
    {
        return "NONE";
    }

    int IntToInt( int value )
    {
        return 0;
    }

    int UIntToInt( unsigned int value )
    {
        return 0;
    }

    int StringToInt( std::string value )
    {
        return 0;
    }

    int IntFromInt( int value )
    {
        return 0;
    }

    unsigned int UIntFromInt( int value )
    {
        return 0U;
    }

    std::string StringFromInt( int value )
    {
        return "0";
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
    const eType TypeInfo<T>::StringToTypeCode( std::string typeText )
    {
        if( typeText == "int"  ) return TYPE_INT;
        if( typeText == "real" ) return TYPE_REAL;
        if( typeText == "text" ) return TYPE_TEXT;
        if( typeText == "ext"  ) return TYPE_EXTENDED;

        return TYPE_NON;
    }

    // Fundamental types
    REGISTER_TYPE( std::string    , "text" , 4, false );
    REGISTER_TYPE( int            , "int"  , 4, true  );
    REGISTER_TYPE( unsigned int   , "int"  , 4, false );
    REGISTER_TYPE( short          , "int"  , 4, true  );
    REGISTER_TYPE( unsigned short , "int"  , 4, false );
    REGISTER_TYPE( float          , "real" , 4, true  );
    REGISTER_TYPE( double         , "real" , 4, true  );

    void CODEFRAME_TYPES_INITIALIZE( void )
    {
        GetTypeInfo<int         >().SetFromStringCallback( &IntFromString    );
        GetTypeInfo<unsigned int>().SetFromStringCallback( &UIntFromString   );
        GetTypeInfo<std::string >().SetFromStringCallback( &StringFromString );

        GetTypeInfo<int         >().SetToStringCallback( &IntToString    );
        GetTypeInfo<unsigned int>().SetToStringCallback( &UIntToString   );
        GetTypeInfo<std::string >().SetToStringCallback( &StringToString );

        GetTypeInfo<int         >().SetFromIntegerCallback( &IntFromInt    );
        GetTypeInfo<unsigned int>().SetFromIntegerCallback( &UIntFromInt   );
        GetTypeInfo<std::string >().SetFromIntegerCallback( &StringFromInt );

        GetTypeInfo<int         >().SetToIntegerCallback( &IntToInt    );
        GetTypeInfo<unsigned int>().SetToIntegerCallback( &UIntToInt   );
        GetTypeInfo<std::string >().SetToIntegerCallback( &StringToInt );

        GetTypeInfo<int         >().SetFromRealCallback( &IntFromReal    );
        GetTypeInfo<unsigned int>().SetFromRealCallback( &UIntFromReal   );
        GetTypeInfo<std::string >().SetFromRealCallback( &StringFromReal );

        GetTypeInfo<int         >().SetToRealCallback( &IntToReal    );
        GetTypeInfo<unsigned int>().SetToRealCallback( &UIntToReal   );
        GetTypeInfo<std::string >().SetToRealCallback( &StringToReal );
    }
}
