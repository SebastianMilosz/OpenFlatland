#include "constelementsfactory.hpp"

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

}
