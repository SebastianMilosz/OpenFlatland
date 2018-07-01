#include "typeinfo.hpp"

#include <MathUtilities.h>

namespace codeframe
{
    int StringToInteger( void* value )
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
        if( typeText == "text" ) return TYPE_TEXT;

        return TYPE_NON;
    }

    REGISTER_TYPE( std::string  , "text"    );
    REGISTER_TYPE( int          , "int"     );
    REGISTER_TYPE( unsigned int , "int"     );

    void CODEFRAME_TYPES_INITIALIZE( void )
    {
        GetTypeInfo<std::string>().SetToIntegerCallback( &StringToInteger );
    }
}
