#ifndef CXML_H_INCLUDED
#define CXML_H_INCLUDED

#include <string>

#include "cxmlnode.hpp"

namespace pugi
{
    class xml_document;
}

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief
      * @author Sebastian Milosz
      * @version 1.0
     **
    ******************************************************************************/
    class cXML
    {
    public:
                 cXML();
                 cXML( const cXML& xml      );
        explicit cXML( const std::string& filePath );
        explicit cXML( cXMLNode xmlNode     );
        explicit cXML( const char* data, int dsize );
       ~cXML();

        cXML&         PointToRoot(                              );
        cXML&         PointToNode( const std::string& name      );
        cXML&         Add        ( cXML& xml                    );

        std::string   Protocol   (                              );
        std::string   ToString   (                              );
        void          ToFile     ( const std::string& filePath  );
        void          FromFile   ( const std::string& filePath  );
        void          FromString ( const std::string& xmlString );
        void          FromBuffer ( const char* data, int dsize  );
        void          Dispose    (                              );
        bool          IsValid() const;
        cXMLNode      GetChild( const std::string& name ) const;
        cXMLNode      FindChildByAttribute( const char_t* name, const char_t* attr_name, const char_t* attr_value ) const;
        cXMLNode      Root();
        cXMLNode      AppendChild(const char_t* name);
        cXMLNode      FirstChild();

        // Operatory
        cXML& operator=(cXML& arg);  // copy/move

    private:
        void Initialize( void );
        void CreateXMLDeclaration( void );

        bool                m_valid;
        pugi::xml_document* m_xmlDocument;
        std::string         m_nodePointer;
    };

}

#endif // CXML_H_INCLUDED
