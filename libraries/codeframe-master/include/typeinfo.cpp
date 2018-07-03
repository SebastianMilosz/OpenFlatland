#include "typeinfo.hpp"

#include <MathUtilities.h>

namespace codeframe
{
    int StringToInteger( void* value, unsigned char bytePrec, bool sign )
    {
        std::string* stringTypePtr = static_cast<std::string*>(value);

        if( NULL != stringTypePtr )
        {
            return utilities::math::StrToInt( *stringTypePtr );
        }

        return 0;
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
        GetTypeInfo<std::string>().SetToIntegerCallback( &StringToInteger );
    }
}
