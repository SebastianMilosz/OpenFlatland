#include "serializable_path.hpp"

#include <vector>
#include <TextUtilities.h>
#include <LoggerUtilities.h>

#include "serializable_object_node.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPath::cPath( ObjectNode& sint ) :
        m_sint( sint ),
        m_parent( NULL )
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPath::~cPath()
    {

    }

    /*****************************************************************************/
    /**
      * @brief Return full object path
     **
    ******************************************************************************/
    std::string cPath::PathString() const
    {
        std::string path;

        ObjectNode* parent = m_sint.Path().Parent();

        if( (ObjectNode*)NULL != parent )
        {
           path = parent->Path().PathString() + "/" + path;
        }

        path += m_sint.Identity().ObjectName( true );

        return path ;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPath::ParentBound( ObjectNode* parent )
    {
        // Rejestrujemy sie u rodzica
        if( parent )
        {
            m_parent = parent;
            m_parent->ChildList().Register( &m_sint );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cPath::ParentUnbound()
    {
        if( m_parent )
        {
            m_parent->ChildList().UnRegister( &m_sint );
            m_parent = NULL;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cPath::IsNameUnique( const std::string& name, const bool checkParent ) const
    {
        int octcnt = 0;

        // Jesli niema rodzica to jestesmy rootem wiec na tym poziomie jestesmy wyjatkowi
        if( m_parent == NULL )
        {
            return true;
        }

        // Rodzica sprawdzamy tylko na wyrazne zyczenie
        if( checkParent )
        {
            // Sprawdzamy czy rodzic jest wyj¹tkowy
            bool isParentUnique = m_parent->Path().IsNameUnique( m_parent->Identity().ObjectName() );

            // Jesli rodzic nie jest wyjatkowy to dzieci tez nie s¹ wiec niesprawdzamy dalej
            if( isParentUnique == false )
            {
                return false;
            }
        }

        // Jesli rodzic jest wyjatkowy sprawdzamy dzieci
        for( cObjectList::iterator it = m_parent->ChildList().begin(); it != m_parent->ChildList().end(); ++it )
        {
            ObjectNode* iser = *it;

            if( iser )
            {
                if( iser->Identity().ObjectName() == name )
                {
                    octcnt++;
                }
            }
            else
            {
                throw std::runtime_error( "cPath::IsNameUnique() cSerializable* iser = NULL" );
            }
        }

        if(octcnt == 1 ) return true;
        else return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ObjectNode* cPath::Parent() const
    {
        return m_parent;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ObjectNode* cPath::GetChildByName( const std::string& name )
    {
        for ( cObjectList::iterator it = m_sint.ChildList().begin(); it != m_sint.ChildList().end(); ++it )
        {
            ObjectNode* iser = *it;
            std::string objectName = iser->Identity().ObjectName( true );

            if ( objectName == name )
            {
                return iser;
            }
        }
        return NULL;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ObjectNode* cPath::GetRootObject()
    {
        if ( Parent() )
        {
            return Parent()->Path().GetRootObject();
        }

        return &m_sint;
    }

    /*****************************************************************************/
    /**
      * @brief Return serializable object from string path
     **
    ******************************************************************************/
    ObjectNode* cPath::GetObjectFromPath( const std::string& path )
    {
        // Rozdzelamy stringa na kawalki
        std::vector<std::string> tokens;
        std::string              delimiters = "/";
        ObjectNode*  curObject = &m_sint;

        utilities::text::split( path, delimiters, tokens);

        // Sprawdzamy czy root sie zgadza
        if ( tokens.size() == 0 )
        {
            if ( curObject->Identity().ObjectName() != path )
            {
                return NULL;
            }
        }
        else
        {
            std::string tempStr = tokens.at(0);
            if( curObject->Identity().ObjectName() != tempStr )
            {
                return NULL;
            }
        }

        // Po wszystkich skladnikach sciezki
        for ( unsigned i = 1; i < tokens.size(); i++ )
        {
            std::string levelName = tokens.at(i);
            curObject = curObject->Path().GetChildByName( levelName );
            if(curObject == NULL) break;
        }

        return curObject;
    }
}
