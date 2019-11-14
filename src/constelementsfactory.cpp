#include "constelementsfactory.hpp"
#include "constelementline.hpp"

#include <extpoint2d.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
ConstElementsFactory::ConstElementsFactory( const std::string& name, ObjectNode* parent ) :
    ObjectContainer( name, parent )
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
smart_ptr<ConstElement> ConstElementsFactory::CreateLine( codeframe::Point2D<int> sPoint, codeframe::Point2D<int> ePoint )
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
smart_ptr<codeframe::ObjectNode> ConstElementsFactory::Create(
                                                     const std::string& className,
                                                     const std::string& objName,
                                                     const std::vector<codeframe::VariantValue>& params
                                                    )
{
    if ( className == "ConstElementLine" )
    {
        codeframe::Point2D<int> startPoint;
        codeframe::Point2D<int> endPoint;

        for ( std::vector<codeframe::VariantValue>::const_iterator it = params.begin(); it != params.end(); ++it )
        {
            if ( it->GetType() == codeframe::TYPE_VECTOR )
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

    return smart_ptr<codeframe::ObjectNode>();
}
