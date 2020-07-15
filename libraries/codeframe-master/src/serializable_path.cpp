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
            path = parentSelection->PathString() + "/" + path;
        }

        path += m_sint.Identity().ObjectName( true );

        return path ;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t cPath::ParentBound( smart_ptr<ObjectNode> parent )
    {
        // Rejestrujemy sie u rodzica
        if( smart_ptr_isValid(parent) )
        {
            m_parent = parent;
            m_parent->ChildList().Register( smart_ptr_wild<ObjectNode>(&m_sint, [](ObjectNode* p) {}) );
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
        if( smart_ptr_isValid(m_parent) )
        {
            m_parent->ChildList().UnRegister( smart_ptr_wild<ObjectNode>(&m_sint, [](ObjectNode* p) {}) );
            m_parent = smart_ptr<ObjectNode>(nullptr);
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t cPath::IsNameUnique( const std::string& name, const bool_t checkParent ) const
    {
        // If there is no parent on this level we are unique
        if ( smart_ptr_isValid(m_parent) == false )
        {
            return true;
        }

        if ( checkParent )
        {
            // Check unique of the parent
            bool_t isParentUnique = m_parent->Path().IsNameUnique( m_parent->Identity().ObjectName() );

            if ( isParentUnique == false )
            {
                return false;
            }
        }

        uint32_t occurrenceCount = 0U;

        for ( auto it = m_sint.ChildList().begin(); it != m_sint.ChildList().end(); ++it )
        {
            smart_ptr<ObjectNode> iser = *it;

            if ( smart_ptr_isValid(iser) )
            {
                if ( iser->Identity().ObjectName(true) == name )
                {
                    occurrenceCount++;
                }
            }
            else
            {
                throw std::runtime_error( "cPath::IsNameUnique() cSerializable* iser = NULL" );
            }
        }

        if ( occurrenceCount == 0U )
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
        if ( smart_ptr_isValid(m_parent) )
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
    smart_ptr<ObjectSelection> cPath::GetRootObject()
    {
        if ( smart_ptr_isValid( Parent() ) )
        {
            return Parent()->Root();
        }

        return smart_ptr<ObjectSelection>( new ObjectSelection( smart_ptr_wild<ObjectNode>(&m_sint, [](ObjectNode* p) {}) ) );
    }

    /*****************************************************************************/
    /**
      * @brief Return object from string path
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> cPath::GetObjectFromPath( const std::string& path )
    {
        auto thisNode = smart_ptr<ObjectSelection>(new ObjectSelection(smart_ptr_wild<ObjectNode>(&m_sint, [](ObjectNode* p) {})));
        cPath::sPathLink pathLink;
        PreparePathLink(path, pathLink, thisNode );

        smart_ptr<ObjectSelection> curObjectSelection = m_sint.Path().GetRootObject();

        if ( smart_ptr_isValid( curObjectSelection ) )
        {
            if ( pathLink.size() )
            {
                if (pathLink.at(0) == curObjectSelection->ObjectName())
                {
                    for ( unsigned int i = 1U; i < pathLink.size(); i++ )
                    {
                        std::string levelName( pathLink.at(i) );

                        curObjectSelection = curObjectSelection->GetObjectByName(levelName);

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

    // With parent we may be able resolve relative paths
    if (smart_ptr_isValid(propertyParent))
    {
        if (IsDownHierarchy(retString))
        {
            smart_ptr<ObjectSelection> parentNode = propertyParent->Parent();

            if (retString.find_first_of("/\\") == 0)
            {
                retString.erase(0, retString.find_first_of("/\\")+1);
            }
            retString.erase(0, retString.find_first_of("/\\"));

            PreparePathLink(retString, pathLink, parentNode );
        }
        else if (IsRelativeHierarchy(retString))
        {
            if (pathLink.size() == 0U)
            {
                std::string pathPropertyString = propertyParent->PathString();
                pathLink.FromDirString(pathPropertyString);

#ifdef CODE_FRAME_REFERENCE_MANAGER_DEBUG
            LOGGER( LOG_INFO << "PreparePathLink: " << pathLink << " pathPropertyString: " << pathPropertyString);
#endif // CODE_FRAME_REFERENCE_MANAGER_DEBUG
            }

            retString.erase(0, retString.find_first_of("/\\")+1);
            pathLink.FromDirString(retString);
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
