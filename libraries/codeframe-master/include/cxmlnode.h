#ifndef CXMLNODE_H_INCLUDED
#define CXMLNODE_H_INCLUDED

typedef char char_t;

namespace pugi
{
    class xml_node;
}

namespace codeframe
{

    class cXMLNode
    {
        friend class cXML;

    public:
         cXMLNode();
         cXMLNode( const cXMLNode& node );
        ~cXMLNode();

         cXMLNode& operator=(cXMLNode node);

        bool IsValid();

        // Find child node by attribute name/value
        cXMLNode FindChildByAttribute(const char_t* name,      const char_t* attr_name, const char_t* attr_value) const;
        cXMLNode FindChildByAttribute(const char_t* attr_name, const char_t* attr_value) const;

        cXMLNode Child          ( const char_t* name) const;
        cXMLNode FirstChild     ( void ) const;
        cXMLNode NextSibling    ( void ) const;
        cXMLNode AppendChild    ( const char_t* name);
        void     AppendAttribute( const char_t* name, const char_t* value);
        void     AppendCopy     ( const cXMLNode& node );

        const char_t* GetAttributeAsString (const char_t* name);
        int           GetAttributeAsInteger(const char_t* name);
        double        GetAttributeAsDouble (const char_t* name);

        const char_t* GetValueAsString( void );
        int           GetValueAsInteger( void );

    private:
        void Initialize();
        cXMLNode( const pugi::xml_node& node );
        pugi::xml_node* m_xmlNode;
    };

}

#endif // CXMLNODE_H_INCLUDED
