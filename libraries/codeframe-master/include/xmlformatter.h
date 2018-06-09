#ifndef XMLFORMATTER_H
#define XMLFORMATTER_H

#include <cxml.h>
#include <vector>
#include <FileUtilities.h>

//#define PATH_FIELD
//#define ID_FIELD

#define XMLTAG_CHILD "child"
#define XMLTAG_OBJECT "obj"
#define XMLTAG_PROPERTY "prop"

namespace codeframe
{
    class cSerializableInterface;

    /*****************************************************************************/
    /**
      * @brief Klasa zapisuje do formatu xml obiekty implementujace interfejs ISerializable
      * @author Sebastian Milosz
      * @version 1.0
     **
    ******************************************************************************/
    class cXmlFormatter
    {
    public:
        cXmlFormatter( cSerializableInterface* serializableObject, int shareLevel = 1 ); ///< Tworzymy formater z obiektu, domyslnie pelna rekurencyjna serializacja
       ~cXmlFormatter();

        cXML           SaveToXML  ();               ///< Zwraca xml z powiazanego obiektu
        cXmlFormatter& LoadFromXML( cXML& xml );    ///< Przypisuje xml z kontenera o nazwie name do powiazanego obiektu

    private:
        cSerializableInterface* m_serializableObject;
        int                     m_shareLevel;

        cXmlFormatter& LoadFromXML_v0( cXML& xml );
        cXmlFormatter& LoadFromXML_v1( cXML& xml );

        cXMLNode FindFirstByAttribute( const cXMLNode& xmlTop, const char_t* name_, const char_t* attr_name, const char_t* attr_value);

        void        ReplaceAll(std::string& str, const std::string& old, const std::string& repl);
        std::string FromEscapeXml( std::string str );
    };

}

#endif // XMLFORMATTER_H
