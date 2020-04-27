#include "serializable_path.hpp"

#include <cstring>      // std::strlen
#include <vector>
#include <algorithm>    // std::reverse
#include <TextUtilities.h>
#include <LoggerUtilities.h>

#include "serializable_object_node.hpp"

namespace codeframe
{
    const std::string cPath::m_delimiters("/\\");

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPath::cPath( ObjectNode& sint ) :
        m_sint( sint ),
        m_parent( nullptr )
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

        smart_ptr<ObjectSelection> parentSelection = m_sint.Path().Parent();

        if( smart_ptr_isValid( parentSelection ) )
        {
            path = parentSelection->GetNode()->Path().PathString() + "/" + path;
        }

        path += m_sint.Identity().ObjectName( true );

        return path ;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t cPath::ParentBound( ObjectNode* parent )
    {
        // Rejestrujemy sie u rodzica
        if( parent )
        {
            m_parent = parent;
            m_parent->ChildList().Register( &m_sint );
            return true;
        }
        return false;
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
            m_parent = nullptr;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t cPath::IsNameUnique( const std::string& name, const bool_t checkParent ) const
    {
        int octcnt = 0;

        // If there is no parent on this level we are unique
        if ( m_parent == nullptr )
        {
            return true;
        }

        if ( checkParent )
        {
            // Check unique of the parent
            bool_t isParentUnique = m_parent->Path().IsNameUnique( m_parent->Identity().ObjectName() );

            if( isParentUnique == false )
            {
                return false;
            }
        }

        for ( auto it = m_parent->ChildList().begin(); it != m_parent->ChildList().end(); ++it )
        {
            ObjectNode* iser = *it;

            if ( iser != nullptr )
            {
                if ( iser->Identity().ObjectName() == name )
                {
                    octcnt++;
                }
            }
            else
            {
                throw std::runtime_error( "cPath::IsNameUnique() cSerializable* iser = NULL" );
            }
        }

        if ( octcnt == 1 )
        {
            return true;
        }

        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> cPath::Parent() const
    {
        if ( m_parent != nullptr )
        {
            return smart_ptr<ObjectSelection>( new ObjectSelection(m_parent) );
        }

        return smart_ptr<ObjectSelection>( nullptr );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> cPath::GetChildByName( const std::string& name )
    {
        // Separate * symbol
        std::size_t foundRangeOpen  = name.find_last_of("[");

        // Multi selection
        if ( foundRangeOpen != std::string::npos )
        {
            if ( name.at( foundRangeOpen + 1U ) == '*' )
            {
                auto  nameCore( name.substr( 0, foundRangeOpen + 1U ) );

                smart_ptr<ObjectMultipleSelection> multipleSelection = smart_ptr<ObjectMultipleSelection>( new ObjectMultipleSelection );

                for ( auto it = m_sint.ChildList().begin(); it != m_sint.ChildList().end(); ++it )
                {
                    ObjectNode* iser = *it;
                    auto  objectName( iser->Identity().ObjectName( true ) );
                    auto  refName( nameCore + ( std::to_string( (int)it ).append( "]" ) ) );

                    if ( objectName == refName )
                    {
                        multipleSelection->Add( iser );
                    }
                }

                return multipleSelection;
            }
        }

        // Single selection
        for ( auto it = m_sint.ChildList().begin(); it != m_sint.ChildList().end(); ++it )
        {
            ObjectNode* iser = *it;
            auto objectName( iser->Identity().ObjectName( true ) );

            if ( objectName == name )
            {
                return smart_ptr<ObjectSelection>( new ObjectSelection( iser ) );
            }
        }
        return smart_ptr<ObjectSelection>(  nullptr );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> cPath::GetRootObject()
    {
        if ( smart_ptr_isValid( Parent() ) )
        {
            return Parent()->GetNode()->Path().GetRootObject();
        }

        return smart_ptr<ObjectSelection>( new ObjectSelection( &m_sint ) );
    }

    /*****************************************************************************/
    /**
      * @brief Return serializable object from string path
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> cPath::GetObjectFromPath( const std::string& path )
    {
        auto thisNode = smart_ptr<ObjectSelection>(new ObjectSelection(&m_sint));
        std::vector<sPathNode> pathDir;
        PreparePath(path, pathDir, thisNode );

        smart_ptr<ObjectSelection> curObjectSelection = smart_ptr<ObjectSelection>( new ObjectSelection( &m_sint ) );

        LOGGER( LOG_INFO << "GetObjectFromPath: " << path << " pathDir: " << pathDir << " this path: " << m_sint.Path().PathString() );

        if ( pathDir.size() )
        {
            std::string tempStr( pathDir.at(0) );
            std::string objectRootName(curObjectSelection->GetNode()->Path().GetRootObject()->GetNode()->Identity().ObjectName());
            if ( objectRootName != tempStr )
            {
                return smart_ptr<ObjectSelection>( nullptr );
            }
        }

        smart_ptr<ObjectSelection> objectSelection = curObjectSelection->GetNode()->Path().Parent();

        // Po wszystkich skladnikach sciezki
        for ( unsigned int i = 1U; i < pathDir.size(); i++ )
        {
            std::string levelName( pathDir.at(i) );
            std::string objectRootName;

            if ( smart_ptr_isValid( objectSelection ) )
            {
                curObjectSelection = objectSelection;
            }
            else
            {
                return smart_ptr<ObjectSelection>( nullptr );
            }
        }

        return objectSelection;
    }

/*****************************************************************************/
/**
  * @brief This method change relative paths to absolute ones
 **
******************************************************************************/
void cPath::PreparePath( const std::string& path, std::vector<sPathNode>& pathDir, smart_ptr<ObjectSelection> propertyParent )
{
    std::string retString( path );

    // With parent we may be able resolve relative path
    if (propertyParent)
    {
        if (IsDownHierarchy(retString))
        {
            smart_ptr<ObjectSelection> parentNode = propertyParent->GetNode()->Path().Parent();
            if (parentNode)
            {
                pathDir.push_back( sPathNode(parentNode->GetNode()->Identity().ObjectName()));
            }

            if (retString.find_first_of("/\\") == 0)
            {
                retString.erase(0, retString.find_first_of("/\\")+1);
            }
            retString.erase(0, retString.find_first_of("/\\"));
            smart_ptr<ObjectSelection> parentObject = propertyParent->GetNode()->Path().Parent();
            PreparePath(retString, pathDir, parentObject );
        }
        else if (IsRelativeHierarchy(retString))
        {
            retString.erase(0, retString.find_first_of("/\\")+1);

            std::string pathDirString;
            for (auto it = pathDir.rbegin(); it != pathDir.rend(); ++it)
            {
                pathDirString += *it;
                pathDirString += "/";
            }

            std::reverse(std::begin(pathDir), std::end(pathDir));
            pathDir.push_back(retString);

            pathDirString.pop_back();   // Remove last / char
            retString = pathDirString + std::string("/") + retString;
        }
        else
        {
            utilities::text::split( path, m_delimiters, pathDir);
        }
    }

    //return retString;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool_t cPath::IsDownHierarchy(const std::string& path)
{
    bool_t retVal = (strncmp(path.c_str(), "/..", std::strlen("/..")) == 0);
    retVal |= (strncmp(path.c_str(), "..", std::strlen("..")) == 0);
    retVal |= (strncmp(path.c_str(), "\\..", std::strlen("\\..")) == 0);
    return retVal;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool_t cPath::IsRelativeHierarchy(const std::string& path)
{
    return (path.find_first_of("/\\") == 0U);
}
}
