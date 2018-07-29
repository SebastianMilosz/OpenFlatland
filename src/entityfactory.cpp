#include "entityfactory.h"

#include <typeinfo.hpp>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::EntityFactory( World& world ) :
    cSerializableContainer( "EntityFactory", NULL ),
    m_world( world )
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
EntityFactory::~EntityFactory()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<Entity> EntityFactory::Create( int x, int y, int z )
{
    smart_ptr<Entity> entity = smart_ptr<Entity>( new Entity( "Unknown", x, y, z ) );

    InsertObject( entity );

    return entity;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
smart_ptr<codeframe::cSerializableInterface> EntityFactory::Create(
                                                                   const std::string className,
                                                                   const std::string objName,
                                                                   const std::vector<codeframe::VariantValue>& params )
{
    if ( className == "Entity" )
    {
        int x = 0;
        int y = 0;
        int z = 0;

        for ( std::vector<codeframe::VariantValue>::const_iterator it = params.begin(); it != params.end(); ++it )
        {
            if ( it->Type == codeframe::TYPE_INT )
            {
                     if ( it->Name == "X" )
                {
                    x = utilities::math::StrToInt( it->Value );
                }
                else if ( it->Name == "Y" )
                {
                    y = utilities::math::StrToInt( it->Value );
                }
                else if ( it->Name == "Z" )
                {
                    z = utilities::math::StrToInt( it->Value );
                }
            }
        }

        smart_ptr<codeframe::cSerializableInterface> obj = smart_ptr<codeframe::cSerializableInterface>( new Entity( objName, x, y, z ) );

        InsertObject( obj );
        return obj;
    }

    return smart_ptr<codeframe::cSerializableInterface>();
}
