#include "serializable_path.hpp"

#include <cstring>      // std::strlen
#include <vector>

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
        cPath::sPathLink pathLink;
        PreparePathLink(path, pathLink, thisNode );

        smart_ptr<ObjectSelection> curObjectSelection = m_sint.Path().GetRootObject();

        LOGGER( LOG_INFO << "GetObjectFromPath: " << path << " pathLink: " << pathLink << " this path: " << m_sint.Path().PathString() );

        if ( smart_ptr_isValid( curObjectSelection ) )
        {
            if ( pathLink.size() )
            {
                // check root match
                if (pathLink.at(0) == curObjectSelection->GetNode()->Identity().ObjectName())
                {
                    for ( unsigned int i = 1U; i < pathLink.size(); i++ )
                    {
                        std::string levelName( pathLink.at(i) );
                        curObjectSelection = curObjectSelection->GetNode()->Path().GetChildByName(levelName);

                        if ( smart_ptr_isValid( curObjectSelection ) == false )
                        {
                            curObjectSelection = smart_ptr<ObjectSelection>( nullptr );
                            break;
                        }
                    }
                }
                else
                {
                    curObjectSelection = smart_ptr<ObjectSelection>( nullptr );
                }
            }
            else
            {
                curObjectSelection = smart_ptr<ObjectSelection>( nullptr );
            }
        }

        return curObjectSelection;
    }

/*****************************************************************************/
/**
  * @brief This method change relative paths to absolute ones
 **
******************************************************************************/
void cPath::PreparePathLink(const std::string& pathString, cPath::sPathLink& pathLink, smart_ptr<ObjectSelection> propertyParent)
{
    std::string retString( pathString );

    // With parent we may be able resolve relative path
    if (propertyParent)
    {
        if (IsDownHierarchy(retString))
        {
            smart_ptr<ObjectSelection> parentNode = propertyParent->GetNode()->Path().Parent();
            if (parentNode)
            {
                pathLink.PathPushBack(parentNode->GetNode()->Identity().ObjectName());
            }

            if (retString.find_first_of("/\\") == 0)
            {
                retString.erase(0, retString.find_first_of("/\\")+1);
            }
            retString.erase(0, retString.find_first_of("/\\"));

            smart_ptr<ObjectSelection> parentObject = propertyParent->GetNode()->Path().Parent();

            if (IsDownHierarchy(retString) == false)
            {
                pathLink.reverse();
            }

            PreparePathLink(retString, pathLink, parentObject );
        }
        else if (IsRelativeHierarchy(retString))
        {
            if (pathLink.size() == 0U)
            {
                pathLink.FromDirString(propertyParent->GetNode()->Path().PathString());
            }

            retString.erase(0, retString.find_first_of("/\\")+1);
            pathLink.PathPushBack(retString);
        }
        else
        {
            pathLink.FromDirString(pathString);
        }
    }
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
