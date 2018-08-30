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

    float FloatFromString( std::string value )
    {
        unsigned int retVal = utilities::math::StrToFloat( value );
        return retVal;
    }

    double DoubleFromString( std::string value )
    {
        unsigned int retVal = utilities::math::StrToDouble( value );
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

    std::vector<float> fVectorFromString( std::string value )
    {

        return std::vector<float>();
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

    std::string FloatToString( const float& value )
    {
        return utilities::math::FloatToStr( value );
    }

    std::string DoubleToString( const double& value )
    {
        return utilities::math::DoubleToStr( value );
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

    IntegerType FloatToInt( const float& value )
    {
        return value;
    }

    IntegerType DoubleToInt( const double& value )
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

    float FloatFromInt( IntegerType value )
    {
        return value;
    }

    double DoubleFromInt( IntegerType value )
    {
        return value;
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

    RealType FloatToReal( const float& value )
    {
        return value;
    }

    RealType DoubleToReal( const double& value )
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

    float FloatFromReal( double value )
    {
        return value;
    }

    double DoubleFromReal( double value )
    {
        return value;
    }

    std::string StringFromReal( double value )
    {
        std::string retVal = std::string("0");
        return retVal;
    }

    bool BoolAddOperator( const bool& value1, const bool& value2 )
    {
        return (value1 + value2);
    }

    int IntAddOperator( const int& value1, const int& value2 )
    {
        return (value1 + value2);
    }

    unsigned int UIntAddOperator( const unsigned int& value1, const unsigned int& value2 )
    {
        return (value1 + value2);
    }

    float FloatAddOperator( const float& value1, const float& value2 )
    {
        return (value1 + value2);
    }

    double DoubleAddOperator( const double& value1, const double& value2 )
    {
        return (value1 + value2);
    }

    std::string StringAddOperator( const std::string& value1, const std::string& value2 )
    {
        return (value1 + value2);
    }

    Point2D Point2AddOperator( const Point2D& value1, const Point2D& value2 )
    {
        return value1;
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
        ToRealCallback( NULL ),
        AddOperatorCallback( NULL )
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

    template<typename T>
    void TypeInfo<T>::SetAddOperatorCallback( T (*toAddOperatorCallback )( const T& value1, const T& value2 ) )
    {
        AddOperatorCallback = toAddOperatorCallback;
    }

    // Fundamental types
    REGISTER_TYPE( std::string       , "text" );
    REGISTER_TYPE( bool              , "int"  );
    REGISTER_TYPE( int               , "int"  );
    REGISTER_TYPE( unsigned int      , "int"  );
    REGISTER_TYPE( short             , "int"  );
    REGISTER_TYPE( unsigned short    , "int"  );
    REGISTER_TYPE( float             , "real" );
    REGISTER_TYPE( double            , "real" );
    REGISTER_TYPE( Point2D           , "vec"  );
    REGISTER_TYPE( std::vector<float>, "vec"  );

    TypeInitializer::TypeInitializer( void )
    {
        GetTypeInfo<bool               >().SetFromStringCallback( &BoolFromString    );
        GetTypeInfo<int                >().SetFromStringCallback( &IntFromString     );
        GetTypeInfo<unsigned int       >().SetFromStringCallback( &UIntFromString    );
        GetTypeInfo<float              >().SetFromStringCallback( &FloatFromString   );
        GetTypeInfo<double             >().SetFromStringCallback( &DoubleFromString  );
        GetTypeInfo<std::string        >().SetFromStringCallback( &StringFromString  );
        GetTypeInfo<Point2D            >().SetFromStringCallback( &Point2DFromString );
        GetTypeInfo<std::vector<float> >().SetFromStringCallback( &fVectorFromString );

        GetTypeInfo<bool        >().SetToStringCallback( &BoolToString    );
        GetTypeInfo<int         >().SetToStringCallback( &IntToString     );
        GetTypeInfo<unsigned int>().SetToStringCallback( &UIntToString    );
        GetTypeInfo<float       >().SetToStringCallback( &FloatToString   );
        GetTypeInfo<double      >().SetToStringCallback( &DoubleToString  );
        GetTypeInfo<std::string >().SetToStringCallback( &StringToString  );
        GetTypeInfo<Point2D     >().SetToStringCallback( &Point2DToString );

        GetTypeInfo<bool        >().SetFromIntegerCallback( &BoolFromInt   );
        GetTypeInfo<int         >().SetFromIntegerCallback( &IntFromInt    );
        GetTypeInfo<unsigned int>().SetFromIntegerCallback( &UIntFromInt   );
        GetTypeInfo<float       >().SetFromIntegerCallback( &FloatFromInt  );
        GetTypeInfo<double      >().SetFromIntegerCallback( &DoubleFromInt );
        GetTypeInfo<std::string >().SetFromIntegerCallback( &StringFromInt );
        GetTypeInfo<Point2D     >().SetFromIntegerCallback( NULL );

        GetTypeInfo<bool        >().SetToIntegerCallback( &BoolToInt    );
        GetTypeInfo<int         >().SetToIntegerCallback( &IntToInt     );
        GetTypeInfo<unsigned int>().SetToIntegerCallback( &UIntToInt    );
        GetTypeInfo<float       >().SetToIntegerCallback( &FloatToInt   );
        GetTypeInfo<double      >().SetToIntegerCallback( &DoubleToInt  );
        GetTypeInfo<std::string >().SetToIntegerCallback( &StringToInt  );
        GetTypeInfo<Point2D     >().SetToIntegerCallback( &Point2DToInt );

        GetTypeInfo<bool        >().SetFromRealCallback( &BoolFromReal   );
        GetTypeInfo<int         >().SetFromRealCallback( &IntFromReal    );
        GetTypeInfo<unsigned int>().SetFromRealCallback( &UIntFromReal   );
        GetTypeInfo<float       >().SetFromRealCallback( &FloatFromReal  );
        GetTypeInfo<double      >().SetFromRealCallback( &DoubleFromReal );
        GetTypeInfo<std::string >().SetFromRealCallback( &StringFromReal );

        GetTypeInfo<bool        >().SetToRealCallback( &BoolToReal   );
        GetTypeInfo<int         >().SetToRealCallback( &IntToReal    );
        GetTypeInfo<unsigned int>().SetToRealCallback( &UIntToReal   );
        GetTypeInfo<float       >().SetToRealCallback( &FloatToReal  );
        GetTypeInfo<double      >().SetToRealCallback( &DoubleToReal );
        GetTypeInfo<std::string >().SetToRealCallback( &StringToReal );

        GetTypeInfo<bool        >().SetAddOperatorCallback( &BoolAddOperator   );
        GetTypeInfo<int         >().SetAddOperatorCallback( &IntAddOperator    );
        GetTypeInfo<unsigned int>().SetAddOperatorCallback( &UIntAddOperator   );
        GetTypeInfo<float       >().SetAddOperatorCallback( &FloatAddOperator  );
        GetTypeInfo<double      >().SetAddOperatorCallback( &DoubleAddOperator );
        GetTypeInfo<std::string >().SetAddOperatorCallback( &StringAddOperator );
        GetTypeInfo<Point2D     >().SetAddOperatorCallback( &Point2AddOperator );
    }
}
