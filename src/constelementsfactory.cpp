#include "constelementsfactory.hpp"
#include "constelementline.hpp"

#include <extendedtypepoint2d.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementsFactory::ConstElementsFactory( std::string name, cSerializableInterface* parent ) :
    cSerializableContainer( name, parent )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementsFactory::~ConstElementsFactory()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ConstElement> ConstElementsFactory::Create( smart_ptr<ConstElement> )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<ConstElement> ConstElementsFactory::CreateLine( codeframe::Point2D sPoint, codeframe::Point2D ePoint )
{
    smart_ptr<ConstElementLine> obj = smart_ptr<ConstElementLine>( new ConstElementLine( "line", sPoint, ePoint ) );

    int id = InsertObject( obj );

    signalElementAdd.Emit( obj );

    return obj;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<codeframe::cSerializableInterface> ConstElementsFactory::Create(
                                                     const std::string className,
                                                     const std::string objName,
                                                     const std::vector<codeframe::VariantValue>& params
                                                    )
{
    if ( className == "ConstElementLine" )
    {
        codeframe::Point2D startPoint;
        codeframe::Point2D endPoint;

        for ( std::vector<codeframe::VariantValue>::const_iterator it = params.begin(); it != params.end(); ++it )
        {
            if ( it->GetType() == codeframe::TYPE_IVECTOR )
            {
                     if ( it->IsName( "SPoint" ) )
                {
                    startPoint.FromStringCallback( it->ValueString );
                }
                else if ( it->IsName( "EPoint" ) )
                {
                    endPoint.FromStringCallback( it->ValueString );
                }
            }
        }

        smart_ptr<ConstElementLine> obj = smart_ptr<ConstElementLine>( new ConstElementLine( objName, startPoint, endPoint ) );

        int id = InsertObject( obj );

        signalElementAdd.Emit( obj );

        return obj;
    }

    return smart_ptr<codeframe::cSerializableInterface>();
}
