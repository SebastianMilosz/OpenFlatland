#include "typeinfo.hpp"

#include <MathUtilities.h>

#include "extvector.hpp"
#include "extfundamental.hpp"

#ifdef USE_RAYDATA_EXT_TYPE
#include <entity_vision_node.hpp>
#endif

#ifdef USE_THRUST_EXT_TYPE
#include "extthrust.hpp"
#endif

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    TypeInfo<T>::TypeInfo( const char* typeName, const char* typeUser, const eType enumType ) :
        TypeCompName( typeName ),
        TypeUserName( typeUser ),
        TypeCode( enumType ),
        FromStringCallback( nullptr ),
        ToStringCallback( nullptr ),
        FromIntegerCallback( nullptr ),
        ToIntegerCallback( nullptr ),
        FromRealCallback( nullptr ),
        ToRealCallback( nullptr ),
        AddOperatorCallback( nullptr )
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    const eType TypeInfo<T>::StringToTypeCode( const std::string& typeText )
    {
        if ( typeText == "int"  ) return TYPE_INT;
        if ( typeText == "real" ) return TYPE_REAL;
        if ( typeText == "text" ) return TYPE_TEXT;
        if ( typeText == "vec"  ) return TYPE_VECTOR;

        return TYPE_NON;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    void TypeInfo<T>::SetFromStringCallback( T (*fromStringCallback)( const StringType& value ) )
    {
        FromStringCallback = fromStringCallback;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    void TypeInfo<T>::SetToStringCallback( StringType (*toStringCallback)( const T& value ) )
    {
        ToStringCallback = toStringCallback;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    void TypeInfo<T>::SetFromIntegerCallback( T (*fromIntegerCallback)( const IntegerType& value ) )
    {
        FromIntegerCallback = fromIntegerCallback;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    void TypeInfo<T>::SetToIntegerCallback( IntegerType (*toIntegerCallback)( const T& value ) )
    {
        ToIntegerCallback = toIntegerCallback;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    void TypeInfo<T>::SetFromRealCallback( T (*fromRealCallback)( const RealType& value ) )
    {
        FromRealCallback = fromRealCallback;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    void TypeInfo<T>::SetToRealCallback( RealType (*toRealCallback)( const T& value ) )
    {
        ToRealCallback = toRealCallback;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    template<typename T>
    void TypeInfo<T>::SetAddOperatorCallback( T (*toAddOperatorCallback )( const T& value1, const T& value2 ) )
    {
        AddOperatorCallback = toAddOperatorCallback;
    }

    // Fundamental types
    REGISTER_TYPE( std::string                      , "text" );
    REGISTER_TYPE( bool                             , "int"  );
    REGISTER_TYPE( int                              , "int"  );
    REGISTER_TYPE( unsigned int                     , "int"  );
    REGISTER_TYPE( short                            , "int"  );
    REGISTER_TYPE( unsigned short                   , "int"  );
    REGISTER_TYPE( float                            , "real" );
    REGISTER_TYPE( double                           , "real" );
    REGISTER_TYPE( Point2D<int>                     , "vec"  );
    REGISTER_TYPE( Point2D<unsigned int>            , "vec"  );
    REGISTER_TYPE( Point2D<float>                   , "vec"  );
    REGISTER_TYPE( Point2D<double>                  , "vec"  );
    REGISTER_TYPE( std::vector<float>               , "vec"  );
    REGISTER_TYPE( std::vector<unsigned int>        , "vec"  );
#ifdef USE_RAYDATA_EXT_TYPE
    REGISTER_TYPE( std::vector<RayData>             , "vec"  );
#endif
#ifdef USE_THRUST_EXT_TYPE
    REGISTER_TYPE( thrust::host_vector<float>       , "vec"  );
    REGISTER_TYPE( thrust::host_vector<unsigned int>, "vec"  );
    REGISTER_TYPE( thrust::host_vector<RayData>     , "vec"  );
#endif

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    TypeInitializer::TypeInitializer( void )
    {
        GetTypeInfo<bool                  >().SetFromStringCallback( &FundamentalTypes<bool>::BoolFromString    );
        GetTypeInfo<int                   >().SetFromStringCallback( &FundamentalTypes<int>::IntFromString     );
        GetTypeInfo<unsigned int          >().SetFromStringCallback( &FundamentalTypes<unsigned int>::UIntFromString    );
        GetTypeInfo<float                 >().SetFromStringCallback( &FundamentalTypes<float>::FloatFromString   );
        GetTypeInfo<double                >().SetFromStringCallback( &FundamentalTypes<double>::DoubleFromString  );
        GetTypeInfo<std::string           >().SetFromStringCallback( &FundamentalTypes<std::string>::StringFromString  );
        GetTypeInfo<Point2D<unsigned int> >().SetFromStringCallback( &Point2D<unsigned int>::Point2DFromString );
        GetTypeInfo<Point2D<int>          >().SetFromStringCallback( &Point2D<int>::Point2DFromString );
        GetTypeInfo<Point2D<float>        >().SetFromStringCallback( &Point2D<float>::Point2DFromString );
        GetTypeInfo<Point2D<double>       >().SetFromStringCallback( &Point2D<double>::Point2DFromString );
        GetTypeInfo<std::vector<float>    >().SetFromStringCallback( &PropertyVector<float>::VectorFromString );
        GetTypeInfo<std::vector<unsigned int> >().SetFromStringCallback( &PropertyVector<unsigned int>::VectorFromString );
#ifdef USE_RAYDATA_EXT_TYPE
        GetTypeInfo<std::vector<RayData>  >().SetFromStringCallback( &PropertyVector<RayData>::VectorFromString );
#endif
#ifdef USE_THRUST_EXT_TYPE
        GetTypeInfo<thrust::host_vector<float> >().SetFromStringCallback( &PropertyThrustVector<float>::VectorFromString );
        GetTypeInfo<thrust::host_vector<unsigned int> >().SetFromStringCallback( &PropertyThrustVector<unsigned int>::VectorFromString );
        GetTypeInfo<thrust::host_vector<RayData> >().SetFromStringCallback( &PropertyThrustVector<RayData>::VectorFromString );
#endif

        GetTypeInfo<bool                  >().SetToStringCallback( &FundamentalTypes<bool>::BoolToString    );
        GetTypeInfo<int                   >().SetToStringCallback( &FundamentalTypes<int>::IntToString     );
        GetTypeInfo<unsigned int          >().SetToStringCallback( &FundamentalTypes<unsigned int>::UIntToString    );
        GetTypeInfo<float                 >().SetToStringCallback( &FundamentalTypes<float>::FloatToString   );
        GetTypeInfo<double                >().SetToStringCallback( &FundamentalTypes<double>::DoubleToString  );
        GetTypeInfo<std::string           >().SetToStringCallback( &FundamentalTypes<std::string>::StringToString  );
        GetTypeInfo<Point2D<unsigned int> >().SetToStringCallback( &Point2D<unsigned int>::Point2DToString );
        GetTypeInfo<Point2D<int>          >().SetToStringCallback( &Point2D<int>::Point2DToString );
        GetTypeInfo<Point2D<float>        >().SetToStringCallback( &Point2D<float>::Point2DToString );
        GetTypeInfo<Point2D<double>       >().SetToStringCallback( &Point2D<double>::Point2DToString );
        GetTypeInfo<std::vector<float>    >().SetToStringCallback( &PropertyVector<float>::VectorToString );
        GetTypeInfo<std::vector<unsigned int> >().SetToStringCallback( &PropertyVector<unsigned int>::VectorToString );
#ifdef USE_RAYDATA_EXT_TYPE
        GetTypeInfo<std::vector<RayData>  >().SetToStringCallback( &PropertyVector<RayData>::VectorToString );
#endif
#ifdef USE_THRUST_EXT_TYPE
        GetTypeInfo<thrust::host_vector<float> >().SetToStringCallback( &PropertyThrustVector<float>::VectorToString );
        GetTypeInfo<thrust::host_vector<unsigned int> >().SetToStringCallback( &PropertyThrustVector<unsigned int>::VectorToString );
        GetTypeInfo<thrust::host_vector<RayData> >().SetToStringCallback( &PropertyThrustVector<RayData>::VectorToString );
#endif

        GetTypeInfo<bool                  >().SetFromIntegerCallback( &FundamentalTypes<bool>::BoolFromInt   );
        GetTypeInfo<int                   >().SetFromIntegerCallback( &FundamentalTypes<int>::IntFromInt    );
        GetTypeInfo<unsigned int          >().SetFromIntegerCallback( &FundamentalTypes<unsigned int>::UIntFromInt   );
        GetTypeInfo<float                 >().SetFromIntegerCallback( &FundamentalTypes<float>::FloatFromInt  );
        GetTypeInfo<double                >().SetFromIntegerCallback( &FundamentalTypes<double>::DoubleFromInt );
        GetTypeInfo<std::string           >().SetFromIntegerCallback( &FundamentalTypes<std::string>::StringFromInt );
        GetTypeInfo<Point2D<unsigned int> >().SetFromIntegerCallback( nullptr );
        GetTypeInfo<Point2D<int>          >().SetFromIntegerCallback( nullptr );
        GetTypeInfo<Point2D<float>        >().SetFromIntegerCallback( nullptr );
        GetTypeInfo<Point2D<double>       >().SetFromIntegerCallback( nullptr );

        GetTypeInfo<bool                  >().SetToIntegerCallback( &FundamentalTypes<bool>::BoolToInt    );
        GetTypeInfo<int                   >().SetToIntegerCallback( &FundamentalTypes<int>::IntToInt     );
        GetTypeInfo<unsigned int          >().SetToIntegerCallback( &FundamentalTypes<unsigned int>::UIntToInt    );
        GetTypeInfo<float                 >().SetToIntegerCallback( &FundamentalTypes<float>::FloatToInt   );
        GetTypeInfo<double                >().SetToIntegerCallback( &FundamentalTypes<double>::DoubleToInt  );
        GetTypeInfo<std::string           >().SetToIntegerCallback( &FundamentalTypes<std::string>::StringToInt  );
        GetTypeInfo<Point2D<unsigned int> >().SetToIntegerCallback( &Point2D<unsigned int>::Point2DToInt );
        GetTypeInfo<Point2D<int>          >().SetToIntegerCallback( &Point2D<int>::Point2DToInt );
        GetTypeInfo<Point2D<float>        >().SetToIntegerCallback( &Point2D<float>::Point2DToInt );
        GetTypeInfo<Point2D<double>       >().SetToIntegerCallback( &Point2D<double>::Point2DToInt );

        GetTypeInfo<bool        >().SetFromRealCallback( &FundamentalTypes<bool>::BoolFromReal   );
        GetTypeInfo<int         >().SetFromRealCallback( &FundamentalTypes<int>::IntFromReal    );
        GetTypeInfo<unsigned int>().SetFromRealCallback( &FundamentalTypes<unsigned int>::UIntFromReal   );
        GetTypeInfo<float       >().SetFromRealCallback( &FundamentalTypes<float>::FloatFromReal  );
        GetTypeInfo<double      >().SetFromRealCallback( &FundamentalTypes<double>::DoubleFromReal );
        GetTypeInfo<std::string >().SetFromRealCallback( &FundamentalTypes<std::string>::StringFromReal );

        GetTypeInfo<bool        >().SetToRealCallback( &FundamentalTypes<bool>::BoolToReal   );
        GetTypeInfo<int         >().SetToRealCallback( &FundamentalTypes<int>::IntToReal    );
        GetTypeInfo<unsigned int>().SetToRealCallback( &FundamentalTypes<unsigned int>::UIntToReal   );
        GetTypeInfo<float       >().SetToRealCallback( &FundamentalTypes<float>::FloatToReal  );
        GetTypeInfo<double      >().SetToRealCallback( &FundamentalTypes<double>::DoubleToReal );
        GetTypeInfo<std::string >().SetToRealCallback( &FundamentalTypes<std::string>::StringToReal );

        GetTypeInfo<bool                  >().SetAddOperatorCallback( &FundamentalTypes<bool>::BoolAddOperator   );
        GetTypeInfo<int                   >().SetAddOperatorCallback( &FundamentalTypes<int>::IntAddOperator    );
        GetTypeInfo<unsigned int          >().SetAddOperatorCallback( &FundamentalTypes<unsigned int>::UIntAddOperator   );
        GetTypeInfo<float                 >().SetAddOperatorCallback( &FundamentalTypes<float>::FloatAddOperator  );
        GetTypeInfo<double                >().SetAddOperatorCallback( &FundamentalTypes<double>::DoubleAddOperator );
        GetTypeInfo<std::string           >().SetAddOperatorCallback( &FundamentalTypes<std::string>::StringAddOperator );
        GetTypeInfo<Point2D<unsigned int> >().SetAddOperatorCallback( &Point2D<unsigned int>::Point2AddOperator );
        GetTypeInfo<Point2D<int>          >().SetAddOperatorCallback( &Point2D<int>::Point2AddOperator );
        GetTypeInfo<Point2D<float>        >().SetAddOperatorCallback( &Point2D<float>::Point2AddOperator );
        GetTypeInfo<Point2D<double>       >().SetAddOperatorCallback( &Point2D<double>::Point2AddOperator );
    }
}
