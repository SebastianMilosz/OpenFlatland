#ifndef XMLFORMATTER_H
#define XMLFORMATTER_H

#include <vector>

#include <FileUtilities.h>

#include "cxml.hpp"
#include "typeinfo.hpp"

//#define PATH_FIELD
//#define ID_FIELD

#define XMLTAG_CHILD "child"
#define XMLTAG_OBJECT "obj"
#define XMLTAG_PROPERTY "prop"

namespace codeframe
{
    class ObjectNode;

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    class cXmlFormatter
    {
    public:
        cXmlFormatter( ObjectNode& serializableObject, int shareLevel = 1 ); ///< Tworzymy formater z obiektu, domyslnie pelna rekurencyjna serializacja
       ~cXmlFormatter();

        cXML           SaveToXML  ();               ///< Zwraca xml z powiazanego obiektu
        cXmlFormatter& LoadFromXML( cXML& xml );    ///< Przypisuje xml z kontenera o nazwie name do powiazanego obiektu

    private:
        ObjectNode& m_serializableObject;
        int                     m_shareLevel;

        cXmlFormatter& LoadFromXML_v0( cXML& xml );
        cXmlFormatter& LoadFromXML_v1( cXML& xml );

        cXMLNode FindFirstByAttribute( const cXMLNode& xmlTop, const char_t* name_, const char_t* attr_name, const char_t* attr_value);

        void        ReplaceAll(std::string& str, const std::string& old, const std::string& repl);
        std::string FromEscapeXml( std::string& str );

        void DeserializeObjectProperties( ObjectNode& obj, cXMLNode& node );
        void DeserializeObjectChilds( ObjectNode& obj, cXMLNode& node );
        void ResolveReferences();

        void FillParameterVector( const std::string& buildConstruct, cXMLNode& childNode, std::vector<codeframe::VariantValue>& paramVector );
    };

}

#endif // XMLFORMATTER_H
