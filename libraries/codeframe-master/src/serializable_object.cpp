#include "serializable_object.hpp"

#include <iostream>
#include <exception>
#include <stdexcept>

#include <TextUtilities.h>
#include <LoggerUtilities.h>
#include <xmlformatter.hpp>

using namespace std;

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Object::ConstructPatern() const
    {
        return "";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Object::Object( const std::string& name, ObjectNode* parent ) :
        m_SerializablePath( *this ),
        m_SerializableStorage( *this ),
        m_SerializableSelectable( *this ),
        m_SerializableScript( *this ),
        m_PropertyList( *this ),
        m_Identity( name, *this )
    {
        if (m_SerializablePath.ParentBound( smart_ptr_wild<ObjectNode>(parent, [](ObjectNode* p) {}) ) == true)
        {
            // For containers we resolve references on inserting stage
            if (parent->Role() != codeframe::CONTAINER)
            {
                // Resolve references only at root node
                ReferenceManager::ResolveReferences(*(ObjectNode*)this);
            }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Object::Object( const std::string& name, smart_ptr<ObjectNode> parent ) :
        m_SerializablePath( *this ),
        m_SerializableStorage( *this ),
        m_SerializableSelectable( *this ),
        m_SerializableScript( *this ),
        m_PropertyList( *this ),
        m_Identity( name, *this )
    {
        if (m_SerializablePath.ParentBound( parent ) == true)
        {
            // Resolve references only at root node
            ReferenceManager::ResolveReferences(*(ObjectNode*)this);
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void Object::PulseChanged( bool fullTree )
    {
        m_Identity.EnterPulseState();
        m_PropertyList.PulseChanged();
        m_Identity.LeavePulseState();

        if ( fullTree )
        {
            m_childList.PulseChanged( fullTree );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Object::~Object()
    {
        // Wyrejestrowywujemy sie u rodzica
        m_SerializablePath.ParentUnbound();
        m_PropertyList.ClearPropertyList();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPath& Object::Path()
    {
        return m_SerializablePath;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cStorage& Object::Storage()
    {
        return m_SerializableStorage;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSelectable& Object::Selection()
    {
        return m_SerializableSelectable;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cScript& Object::Script()
    {
        return m_SerializableScript;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cPropertyList& Object::PropertyList()
    {
        return m_PropertyList;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cObjectList& Object::ChildList()
    {
        return m_childList;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cIdentity& Object::Identity()
    {
        return m_Identity;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectNode> Object::Create(
                                    const std::string& className,
                                    const std::string& objName,
                                    const std::vector<codeframe::VariantValue>& params
                                 )
    {
        return smart_ptr<ObjectNode>();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    unsigned int Object::Count() const
    {
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> Object::operator[]( const unsigned int i )
    {
        return m_childList.GetObjectById(i);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> Object::operator[]( const std::string& name )
    {
        return m_childList.GetObjectByName(name);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> Object::Child( const unsigned int i )
    {
        return m_childList.GetObjectById(i);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectSelection> Object::Child( const std::string& name )
    {
        return m_childList.GetObjectByName(name);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<PropertyNode> Object::Property(const std::string& name)
    {
        return m_PropertyList.GetPropertyByName(name);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<PropertyNode> Object::PropertyFromPath(const std::string& path)
    {
        return m_PropertyList.GetPropertyFromPath(path);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Object::ObjectName( bool idSuffix ) const
    {
        return m_Identity.ObjectName(idSuffix);
    }

    /*****************************************************************************/
    /**
      * @brief Zatwierdzenie wszystkich zmian obiektu i jego potomnych
     **
    ******************************************************************************/
    void Object::CommitChanges()
    {
        m_PropertyList.CommitChanges();
        m_childList.CommitChanges();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void Object::Enable( bool val )
    {
        m_PropertyList.Enable( val );
        m_childList.Enable( val );
    }

}
