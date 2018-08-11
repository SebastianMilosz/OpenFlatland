#include "constelementsfactory.hpp"
#include "constelementline.hpp"

#include <extendedtype2dpoint.hpp>

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
            if ( it->GetType() == codeframe::TYPE_INT )
            {
                     if ( it->IsName( "X" ) )
                {
                    //x = it->IntegerValue();
                }
                else if ( it->IsName( "Y" ) )
                {
                    //y = it->IntegerValue();
                }
                else if ( it->IsName( "Z" ) )
                {
                    //z = it->IntegerValue();
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
