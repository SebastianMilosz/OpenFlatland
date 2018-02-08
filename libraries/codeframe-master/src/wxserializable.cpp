#ifdef SERIALIZABLE_USE_WXWIDGETS
#include <wx/propgrid/propgrid.h>
#include <wx/any.h>
#endif

#include <iostream>       // std::cerr
#include <stdexcept>      // std::out_of_range

#include "wxserializable.h"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    wxSerializable::wxSerializable( std::string name, cSerializable* parent ) :
#ifdef SERIALIZABLE_USE_WXWIDGETS
        m_wxPropertyGridPtr(NULL),
#endif
        cSerializable( name, parent )
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void wxSerializable::wxUpdatePropertyValue( Property* prop )
    {
        if( m_wxPropertyGridPtr )
        {
#ifdef SERIALIZABLE_USE_WXWIDGETS
            switch( prop->Info().GetKind() )
            {
                case KIND_LOGIC:
                {
                    m_wxPropertyGridPtr->SetPropertyValue(prop->Name().c_str(), (int)*prop);
                    break;
                }
                case KIND_NUMBER:
                {
                    m_wxPropertyGridPtr->SetPropertyValue(prop->Name().c_str(), (int)*prop);
                    break;
                }
                case KIND_DIR:
                {
                    m_wxPropertyGridPtr->SetPropertyValue(prop->Name().c_str(), wxString(((std::string)*prop).c_str(),  wxConvUTF8));
                    break;
                }
                case KIND_ENUM:
                {
                    m_wxPropertyGridPtr->SetPropertyValue(prop->Name().c_str(), (int)*prop);
                    break;
                }
                default:
                {
                    m_wxPropertyGridPtr->SetPropertyValue(prop->Name().c_str(), (int)*prop);
                }
            }
#else
            (void)prop;
#endif
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void wxSerializable::wxSetPropertyGrid( wxPropertyGrid* propgrid )
    {
        m_wxPropertyGridPtr = propgrid;

#ifdef SERIALIZABLE_USE_WXWIDGETS

        if( propgrid )
        {
            propgrid->Clear();

            // Dodajemy nieedytowalna nazwe obiektu
            wxPGProperty* ObjectNameProperty = new wxStringProperty( "ObjectName", "ObjectName", this->ObjectName() );
            ObjectNameProperty->SetFlagsFromString( "DISABLED" );
            propgrid->Append( ObjectNameProperty );

            // Po wszystkich zadeklarowanych propertisach
            for( unsigned int n = 0; n < m_vMainPropertyList.size(); n++ )
            {
                Property* prop = m_vMainPropertyList.at(n);

                if( prop )
                {
                    // Odpowiedni propertis do rodzaju
                    // Nazwa Propertisa jest dla danego obiektu unikalna więc urzywamy jej do identyfikacji
                    switch( prop->Info().GetKind() )
                    {
                        case KIND_LOGIC:
                        {
                            wxPGProperty* wxprop = propgrid->Append( new wxBoolProperty(prop->Name(), prop->Name(), (int)*prop) );

                            if( prop->Info().GetEnable() == false ) wxprop->SetFlagsFromString( "DISABLED" );

                            break;
                        }
                        case KIND_NUMBER:
                        {
                            wxPGProperty* wxprop = propgrid->Append( new wxIntProperty( prop->Name(), prop->Name(), (int)*prop) );

                            if( prop->Info().GetEnable() == false ) wxprop->SetFlagsFromString( "DISABLED" );

                            break;
                        }
                        case KIND_DIR:
                        {
                            wxPGProperty* wxprop = propgrid->Append( new wxDirProperty( prop->Name(), prop->Name(), (std::string)*prop) );

                            if( prop->Info().GetEnable() == false ) wxprop->SetFlagsFromString( "DISABLED" );

                            break;
                        }
                        case KIND_ENUM:
                        {
                            std::string enumString = prop->Info().GetEnum();

                            if( enumString.size() )
                            {
                                wxArrayInt    arrIds;
                                wxArrayString arrDiet;
                                int cnt = 0;
                                std::stringstream ss( enumString );
                                std::string token;

                                while(std::getline(ss, token, ',')) { arrIds.Add(cnt++); arrDiet.Add(token); }

                                wxPGProperty* wxprop = propgrid->Append( new wxEnumProperty( prop->Name(), prop->Name(), arrDiet, arrIds, (int)*prop ) );

                                if( prop->Info().GetEnable() == false ) wxprop->SetFlagsFromString( "DISABLED" );
                            }
                            else
                            {
                                wxPGProperty* wxprop = propgrid->Append( new wxIntProperty( prop->Name(), prop->Name(), (int)*prop) );

                                if( prop->Info().GetEnable() == false ) wxprop->SetFlagsFromString( "DISABLED" );
                            }
                            break;
                        }
                        default:
                        {
                            wxPGProperty* wxprop = propgrid->Append( new wxStringProperty(prop->Name(), prop->Name(), (std::string)*prop) );

                            if( prop->Info().GetEnable() == false ) wxprop->SetFlagsFromString( "DISABLED" );
                        }
                    }
                }
            }
        }
#endif
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void wxSerializable::wxGetPropertyGrid( wxPropertyGrid* propgrid, bool changedFilter )
    {
        if( propgrid == NULL ) return;

#ifdef SERIALIZABLE_USE_WXWIDGETS

        wxPropertyGridIterator it;

        // Iterating through a property container
        for( it = propgrid->GetIterator(); !it.AtEnd(); it++ )
        {
            wxPGProperty* property = *it;    // Do something with the property

            // @todo dodac sprawdzenie czy zostal zmieniony
            if( property )
            {
                if( (property->GetFlags() & wxPG_PROP_MODIFIED) && changedFilter )
                {
                    std::string propertyName = std::string( property->GetName().c_str() );

                    // Nazwa serializowanego obiektu
                    if( propertyName == "Name" )
                    {
                        wxAny pv = property->GetValue();

                        if( pv.IsNull() == false && pv.CheckType<wxString>() )
                        {
                            std::string propertyValue = pv.As<wxString>().ToStdString();

                            if( this->IsNameUnique( propertyValue ) == false )
                            {
                                LOGGER( LOG_WARNING << "cSerializable::wxGetPropertyGrid IsNameUnique == false" );
                                return;
                            }

                            this->SetName( propertyValue );
                        }
                    }
                    else
                    {
                        // Znajdujemy serializowanego propertisa po nazwie
                        try
                        {
                            Property* serProperty = GetPropertyByName( propertyName );

                            if( serProperty )
                            {
                                // Zaleznie od rodzaju rzutujemy wartosci
                                switch( serProperty->Info().GetKind() )
                                {
                                    case KIND_LOGIC:
                                    {
                                        wxAny pv = property->GetValue();

                                        if( pv.IsNull() == false && pv.CheckType<bool>() )
                                        {
                                            *serProperty = pv.As<bool>();
                                        }
                                        break;
                                    }
                                    case KIND_NUMBER:
                                    {
                                        wxAny pv = property->GetValue();

                                        if( pv.IsNull() == false && pv.CheckType<int>() )
                                        {
                                            *serProperty = pv.As<int>();
                                        }
                                        break;
                                    }
                                    case KIND_DIR:
                                    {
                                        wxAny pv = property->GetValue();

                                        if( pv.IsNull() == false && pv.CheckType<wxString>() )
                                        {
                                            *serProperty = pv.As<wxString>().ToStdString();
                                        }
                                        break;
                                    }
                                    case KIND_ENUM:
                                    {
                                        wxAny pv = property->GetValue();

                                        if( pv.IsNull() == false && pv.CheckType<int>() )
                                        {
                                            *serProperty = pv.As<int>();
                                        }
                                        break;
                                    }
                                    default:
                                    {
                                        wxAny pv = property->GetValue();

                                        if( pv.IsNull() == false && pv.CheckType<wxString>() )
                                        {
                                            *serProperty = pv.As<wxString>().ToStdString();
                                        }
                                    }
                                }
                            }
                        }
                        catch( const std::out_of_range& oor )
                        {
                            LOGGER( LOG_WARNING << "cSerializable::wxGetPropertyGrid out_of_range" );
                        }
                    }
                }
            }
        }
#else
        (void)changedFilter;
#endif
    }
}
