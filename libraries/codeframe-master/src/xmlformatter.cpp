#include "xmlformatter.hpp"

#include <iostream>
#include <exception>
#include <stdexcept>

#include <MathUtilities.h>
#include <LoggerUtilities.h>
#include <TextUtilities.h>

#include "serializableinterface.hpp"
#include "serializablecontainer.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
      * @param filePath
     **
    ******************************************************************************/
    cXmlFormatter::cXmlFormatter( cSerializableInterface* serializableObject, int shareLevel )
    {
        m_shareLevel         = shareLevel;
        m_serializableObject = serializableObject;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cXmlFormatter::~cXmlFormatter()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cXmlFormatter::ReplaceAll(std::string& str, const std::string& old, const std::string& repl)
    {
        size_t pos = 0;
        while ((pos = str.find(old, pos)) != std::string::npos)
        {
            str.replace(pos, old.length(), repl);
            pos += repl.length();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string cXmlFormatter::FromEscapeXml(std::string str)
    {
        ReplaceAll(str, std::string("&amp;" ), std::string("&"));
        ReplaceAll(str, std::string("&apos;"), std::string("'"));
        ReplaceAll(str, std::string("&quot;"), std::string("\""));
        ReplaceAll(str, std::string("&gt;"  ), std::string(">"));
        ReplaceAll(str, std::string("&lt;"  ), std::string("<"));

        return str;
    }

    /*****************************************************************************/
    /**
      * @brief Wersja Load'era zaczytująca stare pliki xml mio-1
     **
    ******************************************************************************/
    cXmlFormatter& cXmlFormatter::LoadFromXML_v0( cXML& xml )
    {
        if( m_serializableObject )
        {
            cXMLNode mio1node = xml.GetChild( std::string("MIO1") );

            // Po wszystkich polach serializacji tego obiektu
            for( cSerializable::iterator it = m_serializableObject->begin(); it != m_serializableObject->end(); ++it )
            {
                PropertyBase* iser = *it;

                cXMLNode propertyNode = mio1node.FindChildByAttribute("feald", "name", iser->Name().c_str());

                // Jesli znaleziono wezel
                if( propertyNode.IsValid() == true )
                {
                    const char_t* type = propertyNode.GetAttributeAsString("type");

                         if( strcmp (type, "int")  == 0 )
                    {
                        *iser = int( propertyNode.GetValueAsInteger() );
                    }
                    else if( strcmp (type, "char") == 0 )
                    {
                        *iser = char( propertyNode.GetValueAsInteger() );
                    }
                    else if( strcmp (type, "text") == 0 )
                    {
                        *iser = std::string( propertyNode.GetValueAsString() );
                    }
                    else
                    {
                        LOGGER( LOG_INFO << "cXmlFormatter::LoadFromXML() Unknown type: " << std::string( type ) );
                    }
                }
            }

            // Deserializacja dzieci
            cXMLNode childNodeContainer = mio1node.FindChildByAttribute("container", "name", "SOCTAB");
            int childLp  = 0;

            // Po wszystkich obiektach dzieci ladujemy zawartosc
            for( cSerializableChildList::iterator it = m_serializableObject->ChildList()->begin(); it != m_serializableObject->ChildList()->end(); ++it )
            {
                cSerializableInterface* iser = *it;

                // Jesli jest to kontener to po jego dzieciach czyli obiektach
                if( iser->Role() == "Container" )
                {
                    // Po wszystkich obiektach dzieci ladujemy zawartosc
                    for( cSerializableChildList::iterator itc = iser->ChildList()->begin(); itc != iser->ChildList()->end(); ++itc )
                    {
                        cXMLNode childNodeElement = childNodeContainer.FindChildByAttribute("element", "id", utilities::math::IntToStr(childLp++).c_str());
                        cXMLNode childNodeObject  = childNodeElement.Child("cSocket");

                        if( childNodeObject.IsValid() == true )
                        {
                            cSerializableInterface* iserc = *itc;

                            // Po wszystkich polach serializacji tego obiektu
                            for( cSerializable::iterator itcp = iserc->begin(); itcp != iserc->end(); ++itcp )
                            {
                                PropertyBase* isercp = *itcp;

                                cXMLNode propertyNode_1 = childNodeObject.FindChildByAttribute("feald", "name", isercp->Name().c_str());

                                // Jesli znaleziono wezel
                                if( propertyNode_1.IsValid() == true )
                                {
                                    const char_t* type = propertyNode_1.GetAttributeAsString("type");

                                         if( strcmp (type, "int")  == 0 )
                                    {
                                        *isercp = int( propertyNode_1.GetValueAsInteger() );
                                    }
                                    else if( strcmp (type, "char") == 0 )
                                    {
                                        *isercp = char( propertyNode_1.GetValueAsInteger() );
                                    }
                                    else if( strcmp (type, "text") == 0 )
                                    {
                                        *isercp = std::string( propertyNode_1.GetValueAsString() );
                                    }
                                    else
                                    {
                                        LOGGER( LOG_INFO << "cXmlFormatter::LoadFromXML() Unknown type: " << std::string( type ) );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cXMLNode cXmlFormatter::FindFirstByAttribute( const cXMLNode& xmlTop, const char_t* name_, const char_t* attr_name, const char_t* attr_value)
    {
        for (cXMLNode child = xmlTop.FirstChild(); child.IsValid(); child = child.NextSibling())
        {
            cXMLNode node = child.FindChildByAttribute(name_, attr_name, attr_value);

            if( node.IsValid() == true )
            {
                return node;
            }

            cXMLNode rootObjNode = FindFirstByAttribute( child, name_, attr_name, attr_value );

            if( rootObjNode.IsValid() == true )
            {
                return rootObjNode;
            }
        }

        return cXMLNode();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cXmlFormatter& cXmlFormatter::LoadFromXML_v1( cXML& xml )
    {
        if( m_serializableObject )
        {
            std::string serializableObjectName = m_serializableObject->ObjectName( false ); // No sufix only name
            int serializableObjectId           = m_serializableObject->GetId();             // container iterator

            // Dozwolone sa tylko nazwy unikalne na danym poziomie
            if( m_serializableObject->IsNameUnique( m_serializableObject->ObjectName() ) == false ) // Test Unique with Id number
            {
                std::string throwString = std::string("cXmlFormatter::LoadFromXML() Name is not Unique: ") + m_serializableObject->ObjectName();

                throw std::runtime_error( throwString );
            }

            cXMLNode rootObjNode;

            if( serializableObjectId >= 0 ) // If Id then unique is Id
            {
                rootObjNode = xml.FindChildByAttribute(XMLTAG_OBJECT, "lp", utilities::math::IntToStr( serializableObjectId ).c_str());

                // Name will have to also match
                std::string name = std::string( rootObjNode.GetAttributeAsString("name") );

                if( name != serializableObjectName )
                {
                    rootObjNode = cXMLNode();
                }
            }
            else // Name is unique
            {
                rootObjNode = xml.FindChildByAttribute(XMLTAG_OBJECT, "name", serializableObjectName.c_str());
            }

            // Jesli nieznaleziono na tym poziomie przeszukujemy glebiej tak dlugo az znajdziemy
            if( rootObjNode.IsValid() == false )
            {
                rootObjNode = FindFirstByAttribute( xml.Root(), XMLTAG_OBJECT, "name", serializableObjectName.c_str());
            }

            // Nie znaleziono w calym drzewie wiec blad i spadamy
            if( rootObjNode.IsValid() == false )
            {
                std::string errormsg = std::string("cXmlFormatter::LoadFromXML() rootObjNode == NULL no object name ");
                errormsg += serializableObjectName;
                errormsg += std::string("inside xml document");


                throw std::runtime_error( errormsg.c_str() );
            }

            // Po wszystkich polach serializacji tego obiektu
            for( cSerializable::iterator it = m_serializableObject->begin(); it != m_serializableObject->end(); ++it )
            {
                PropertyBase* iser = *it;

                if( iser->Info().GetXmlMode() & XMLMODE_R )
                {
                    // Dozwolone sa tylko pola unikalne na danym poziomie
                    if( m_serializableObject->IsPropertyUnique( iser->Name() ) == false )
                    {
                        std::string throwString = std::string("cXmlFormatter::LoadFromXML() PropertyBase is not Unique: ") + iser->Name();

                        throw std::runtime_error( throwString );
                    }

                    cXMLNode propertyNode = rootObjNode.FindChildByAttribute(XMLTAG_PROPERTY, "name", iser->Name().c_str());

                    // Jesli znaleziono wezel
                    if( propertyNode.IsValid() == true )
                    {
                        // Nie wypuszczamy eventow w czasie zaczytywania
                        bool ev = iser->Info().IsEventEnable();
                        iser->Info().Event( false );

                        const char_t* type = propertyNode.GetAttributeAsString("type");

                             if( strcmp (type, "int")  == 0 )
                        {
                            *iser = int   (propertyNode.GetAttributeAsInteger("value") );
                        }
                        else if( strcmp (type, "real") == 0 )
                        {
                            *iser = double( propertyNode.GetAttributeAsDouble("value") );
                        }
                        else if( strcmp (type, "char") == 0 )
                        {
                            *iser = char( propertyNode.GetAttributeAsInteger("value") );
                        }
                        else if( strcmp (type, "text") == 0 )
                        {
                            std::string tempText = std::string(propertyNode.GetAttributeAsString("value") );

                            *iser = FromEscapeXml( tempText );
                        }
                        else if( strcmp (type, "ivec") == 0 )
                        {
                            std::string tempText = std::string(propertyNode.GetAttributeAsString("value") );
                            *iser = tempText;
                        }
                        else if( strcmp (type, "image") == 0 )
                        {

                        }
                        else
                        {
                            LOGGER( LOG_INFO << "cXmlFormatter::LoadFromXML() Unknown type: " << std::string( type ) );
                        }

                        // Zaczytanie enumeracji jesli istnieje
                        const char_t* enumVal = propertyNode.GetAttributeAsString("enum");

                        if( enumVal )
                        {
                            iser->Info().Enum( std::string( enumVal ) );
                        }

                        // Zaczytanie opisu jesli istnieje
                        const char_t* description = propertyNode.GetAttributeAsString("desc");

                        if( description )
                        {
                            std::string tempText = std::string( description );

                            iser->Info().Description( FromEscapeXml( tempText ) );
                        }

                        // Przywracamy stan eventu dla tego propertisa
                        iser->Info().Event( ev );

                        // Sprawdzamy czy jest referencja z innym obiektem
                        std::string href = std::string( propertyNode.GetAttributeAsString("href") );
                        if( href.size() )
                        {
                            // Okreslamy obiekt root dla danego obiektu i wzgledem niego okreslamy cel
                            cSerializableInterface* rootObj = m_serializableObject->GetRootObject();

                            PropertyBase* refProperty = rootObj->GetPropertyFromPath( href );

                            if( refProperty )
                            {
                                if( iser->ConnectReference( refProperty ) == false )
                                {
                                    throw std::runtime_error( "cXmlFormatter::LoadFromXML() Cant create reference" );
                                }
                            }
                            else
                            {
                                throw std::runtime_error( "cXmlFormatter::LoadFromXML() Unresolved reference" );
                            }
                        }

                        // Enable
                        std::string en = std::string( propertyNode.GetAttributeAsString("enable") );
                        if( en.size() )
                        {
                            bool isEnable = char( propertyNode.GetAttributeAsInteger("enable") );
                            iser->Info().Enable( isEnable );
                        }
                    }
                }
            }

            // DeSerializacja dzieci
            std::string thisRole   = std::string(rootObjNode.GetAttributeAsString("role"));
            cXMLNode childNodeContainer = rootObjNode.FindChildByAttribute(XMLTAG_CHILD, "name", m_serializableObject->ObjectName().c_str());
            int childCnt = childNodeContainer.GetAttributeAsInteger("cnt");

            // Jesli rola tego obiektu to kontener obiektow iterujemy po wpisach i sparawdzamy czy nie trzeba
            // czegos dynamicznie stworzyc
            if( thisRole == "Container" )
            {
                // Rzutujemy na kontener
                cSerializableContainer* containerObject = dynamic_cast< cSerializableContainer* >(m_serializableObject);

                // Usuwamy wszystkie dynamiczne obiekty z kontenera
                if( (cSerializableContainer*)NULL != containerObject )
                {
                    cIgnoreList ignore;

                    // Po wszystkich zadeklarowanych dzieciach tworzymy liste ignorowanych przy usuwaniu
                    // Ignorujemy wszystkie elementy które występują w nowej konfiguracji bo one musza tylko zostac zaktualizowane nowymi danymi
                    for( int objectLp = 0; objectLp < childCnt; objectLp++ )
                    {
                        cXMLNode childNode = childNodeContainer.FindChildByAttribute(XMLTAG_OBJECT, "lp", utilities::math::IntToStr( objectLp ).c_str());
                        std::string buildType  = std::string( childNode.GetAttributeAsString("build") );
                        std::string buildClass = std::string( childNode.GetAttributeAsString("class") );
                        std::string buildName  = std::string( childNode.GetAttributeAsString("name") );

                        ignore.AddToList( buildName, buildClass, buildType );
                    }

                    // Pozostawiamy tylko objekty ktore istnieja w nowej konfiguracji kontenera
                    containerObject->DisposeByBuildType( "Dynamic", ignore );

                    // Po wszystkich zadeklarowanych dzieciach dla nowej konfigurscji
                    for( int objectLp = 0; objectLp < childCnt; objectLp++ )
                    {
                        cXMLNode childNode = childNodeContainer.FindChildByAttribute(XMLTAG_OBJECT, "lp", utilities::math::IntToStr( objectLp ).c_str());
                        std::string buildType      = std::string( childNode.GetAttributeAsString("build") );
                        std::string buildClass     = std::string( childNode.GetAttributeAsString("class") );
                        std::string buildName      = std::string( childNode.GetAttributeAsString("name") );
                        std::string buildConstruct = std::string( childNode.GetAttributeAsString("construct") );

                        // Jesli obiekt budowany dynamicznie
                        if( buildType == "Dynamic" )
                        {
                            // Jesli jeszcze nie istnieje dodajemy
                            if( (bool)containerObject->IsName( buildName ) == false )
                            {
                                bool constructRes = false;

                                if( buildConstruct != "" )
                                {
                                    std::vector<std::string> paramNameVector;
                                    utilities::text::split( buildConstruct, ",", paramNameVector);

                                    std::vector<codeframe::VariantValue> paramVector;

                                    // Filling parameter vector
                                    for ( std::vector<std::string>::const_iterator it = paramNameVector.begin(); it != paramNameVector.end(); ++it )
                                    {
                                        std::string propName = *it;

                                        // search for property of the same name
                                        cXMLNode propertyNode = childNode.FindChildByAttribute( XMLTAG_PROPERTY, "name", propName.c_str() );

                                        // Jesli znaleziono wezel
                                        if( propertyNode.IsValid() == true )
                                        {
                                            codeframe::VariantValue variantValue;

                                            variantValue.Name = propName;

                                            const char_t* type = propertyNode.GetAttributeAsString("type");

                                                 if( strcmp (type, "int")  == 0 )
                                            {
                                                variantValue.Type = codeframe::TYPE_INT;
                                                variantValue.Value.Integer = int   (propertyNode.GetAttributeAsInteger("value") );
                                            }
                                            else if( strcmp (type, "real") == 0 )
                                            {
                                                variantValue.Type = codeframe::TYPE_REAL;
                                                variantValue.Value.Real = double( propertyNode.GetAttributeAsDouble("value") );
                                            }
                                            else if( strcmp (type, "char") == 0 )
                                            {
                                                variantValue.Type = codeframe::TYPE_INT;
                                                variantValue.Value.Integer = char( propertyNode.GetAttributeAsInteger("value") );
                                            }
                                            else if( strcmp (type, "text") == 0 )
                                            {
                                                variantValue.Type = codeframe::TYPE_TEXT;
                                                std::string tempText = std::string(propertyNode.GetAttributeAsString("value") );

                                                variantValue.ValueString = FromEscapeXml( tempText );
                                            }
                                            else if( strcmp (type, "ivec") == 0 )
                                            {
                                                variantValue.Type = codeframe::TYPE_IVECTOR;
                                                std::string tempText = std::string(propertyNode.GetAttributeAsString("value") );

                                                variantValue.ValueString = FromEscapeXml( tempText );
                                            }

                                            paramVector.push_back( variantValue );
                                        }
                                    }

                                    constructRes = smart_ptr_isValid( containerObject->Create( buildClass, buildName, paramVector ) );
                                }
                                else
                                {
                                    constructRes = smart_ptr_isValid( containerObject->Create( buildClass, buildName ) );
                                }

                                if( constructRes == false )
                                {
                                    LOGGER( LOG_ERROR  << "Dynamic Object " << buildName << " cannot be created from class: " << buildClass );
                                }
                                else
                                {
                                    LOGGER( LOG_INFO  << "Dynamic Object " << buildName << " created from class: " << buildClass );
                                }
                            }
                        }
                    }
                }
                else
                {
                    LOGGER( LOG_ERROR << "cXmlFormatter::LoadFromXML() dynamic_cast to cSerializableContainer<cSerializable> return NULL" );
                }
            }

            if( m_shareLevel == 1 )
            {
                int childLp  = 0;

                // Po wszystkich obiektach dzieci ladujemy zawartosc
                for( cSerializableChildList::iterator it = m_serializableObject->ChildList()->begin(); it != m_serializableObject->ChildList()->end(); ++it )
                {
                    cSerializableInterface* iser = *it;
                    cXmlFormatter formatter( iser );

                    cXMLNode childNode = childNodeContainer.FindChildByAttribute(XMLTAG_OBJECT, "lp", utilities::math::IntToStr(childLp++).c_str());

                    cXML xml( childNode );

                    formatter.LoadFromXML( xml );
                }
            }
        }
        else
        {
            throw std::runtime_error("cXmlFormatter::LoadFromXML() serializableObject = NULL");
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cXML cXmlFormatter::SaveToXML()
    {
        cXML xmlDocument;

        if( m_serializableObject )
        {
            // Serializacja pol obiektu
            cXMLNode rootNode = xmlDocument.AppendChild( XMLTAG_OBJECT );
            rootNode.AppendAttribute( "name",      m_serializableObject->ObjectName( false ).c_str() );
            rootNode.AppendAttribute( "build",     m_serializableObject->BuildType().c_str() );
            rootNode.AppendAttribute( "role",      m_serializableObject->Role().c_str() );
            rootNode.AppendAttribute( "class",     m_serializableObject->Class().c_str() );
            rootNode.AppendAttribute( "construct", m_serializableObject->ConstructPatern().c_str() );
    #ifdef PATH_FIELD
            rootNode.AppendAttribute("path", m_serializableObject->Path().c_str());
    #endif

            // Po wszystkich polach serializacji
            for( cSerializable::iterator it = m_serializableObject->begin(); it != m_serializableObject->end(); ++it )
            {
                PropertyBase* iser = *it;

                if( iser->Info().GetXmlMode() & XMLMODE_W ) // Jesli dozwolony zapis do xmla dla tego propertisa
                {
                    cXMLNode descr = rootNode.AppendChild(XMLTAG_PROPERTY);

                    descr.AppendAttribute("name", iser->Name().c_str());
                    descr.AppendAttribute("type", iser->TypeString().c_str());

    #ifdef ID_FIELD
                    descr.AppendAttribute("id", LongToHex( iser->Id() ).c_str());
    #endif

                    descr.AppendAttribute("value", ((std::string)(*iser)).c_str());

    #ifdef PATH_FIELD
                    descr.AppendAttribute("path", iser->Path().c_str());
    #endif

                    if( iser->Info().GetDescription() != "" ) descr.AppendAttribute("desc", iser->Info().GetDescription().c_str());
                    if( iser->Info().GetEnum()        != "" ) descr.AppendAttribute("enum", iser->Info().GetEnum().c_str());

                    if( iser->Info().GetKind() != 0  )
                    {
                        eKind kind = iser->Info().GetKind();
                        std::string kindString = utilities::math::IntToStr( static_cast<int>(kind) );
                        descr.AppendAttribute("kind", kindString.c_str() );
                    }

                    if( iser->Info().GetMin() != INT_MIN ) descr.AppendAttribute("min", utilities::math::IntToStr( iser->Info().GetMin() ).c_str());
                    if( iser->Info().GetMax() != INT_MAX ) descr.AppendAttribute("max", utilities::math::IntToStr( iser->Info().GetMax() ).c_str());

                    descr.AppendAttribute("enable", utilities::math::IntToStr( iser->Info().GetEnable() ).c_str());

                    // Doklejamy link do sprzezonego pola jesli takie istnieje
                    if( iser->Reference() != NULL )
                    {
                        descr.AppendAttribute("href", iser->Reference()->Path().c_str());
                    }

                    // Zapis danych Rejestru
                    if( iser->Info().GetRegister().IsEnable() )
                    {
                        eREG_MODE mode = iser->Info().GetRegister().Mode();
                        if( mode == R || mode == RW )
                        {
                            descr.AppendAttribute("rre", utilities::math::IntToHex( iser->Info().GetRegister().ReadRegister()    , "0x", "", -1).c_str());
                            descr.AppendAttribute("rrs", utilities::math::IntToHex( iser->Info().GetRegister().ReadRegisterSize(), "0x", "", -1).c_str());
                            descr.AppendAttribute("rco", utilities::math::IntToHex( iser->Info().GetRegister().ReadCellOffset()  , "0x", "", -1).c_str());
                            descr.AppendAttribute("rcs", utilities::math::IntToHex( iser->Info().GetRegister().ReadCellSize()    , "0x", "", -1).c_str());
                            descr.AppendAttribute("rbm", utilities::math::IntToHex( iser->Info().GetRegister().ReadBitMask()     , "0x", "", -1).c_str());
                        }
                        if( mode == W || mode == RW )
                        {
                            descr.AppendAttribute("wre", utilities::math::IntToHex( iser->Info().GetRegister().WriteRegister()    , "0x", "", -1).c_str());
                            descr.AppendAttribute("wrs", utilities::math::IntToHex( iser->Info().GetRegister().WriteRegisterSize(), "0x", "", -1).c_str());
                            descr.AppendAttribute("wco", utilities::math::IntToHex( iser->Info().GetRegister().WriteCellOffset()  , "0x", "", -1).c_str());
                            descr.AppendAttribute("wcs", utilities::math::IntToHex( iser->Info().GetRegister().WriteCellSize()    , "0x", "", -1).c_str());
                            descr.AppendAttribute("wbm", utilities::math::IntToHex( iser->Info().GetRegister().WriteBitMask()     , "0x", "", -1).c_str());
                        }
                    }
                }
            }

            // Serializacja dzieci
            cXMLNode childNode = rootNode.AppendChild(XMLTAG_CHILD);
            childNode.AppendAttribute("name", m_serializableObject->ObjectName( false ).c_str());
            childNode.AppendAttribute("cnt", utilities::math::IntToStr(m_serializableObject->ChildList()->size()).c_str());

            if( m_shareLevel == 1 )
            {
                int lp = 0;

                for( cSerializableChildList::iterator it = m_serializableObject->ChildList()->begin(); it != m_serializableObject->ChildList()->end(); ++it )
                {
                    cSerializableInterface* iser = *it;

                    if( iser )
                    {
                        cXmlFormatter formatter( iser );

                        cXML childXml( formatter.SaveToXML() );

                        cXMLNode rootObjNode = childXml.FirstChild();
                        rootObjNode.AppendAttribute("lp", utilities::math::IntToStr(lp++).c_str());
                        childNode.AppendCopy( rootObjNode );
                    }
                    else
                    {
                        throw std::runtime_error( "cXmlFormatter::SaveToXML() cSerializable* iser = NULL" );
                    }
                }
            }
        }
        else
        {
            throw std::runtime_error("cXmlFormatter::SaveToXML() serializableObject = NULL");
        }

        return cXML(xmlDocument);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cXmlFormatter& cXmlFormatter::LoadFromXML( cXML& xml )
    {
        try
        {
            return LoadFromXML_v1( xml );
        }
        catch( const std::runtime_error& re )
        {
            LOGGER( LOG_ERROR << "LoadFromXML runtime exception: " << re.what() );
        }
        catch (...)
        {
            LOGGER( LOG_ERROR << "Caught an unknown exception\n" );
        }

        return *this;
    }

}
