#include "typeinfo.hpp"

namespace codeframe
{
    const eType TypeInfo::StringToTypeCode( std::string typeText )
    {
        if( typeText == "text" ) return TYPE_TEXT;

        return TYPE_NON;
    }

    REGISTER_TYPE( std::string  , "text"    );
    REGISTER_TYPE( int          , "int"     );
    REGISTER_TYPE( unsigned int , "int"     );
}
