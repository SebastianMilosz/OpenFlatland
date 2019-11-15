#ifndef CSTORAGE_H
#define CSTORAGE_H

#include "cxml.hpp"

namespace codeframe
{
    class ObjectNode;

    /*****************************************************************************/
    /**
      * @brief This class add storage functionality to cInstanceManager
      * @version 1.0
      * @note Object::cStorage
     **
    ******************************************************************************/
    class cStorage
    {
        public:
            enum eShareLevel
            {
                ShareThis = 0,
                ShareFull
            };

                     cStorage( ObjectNode& sint );
            virtual ~cStorage();

            ObjectNode& ShareLevel  ( eShareLevel level = ShareFull );
            ObjectNode& LoadFromFile( const std::string& filePath, const std::string& container = "", bool createIfNotExist = false );
            ObjectNode& LoadFromXML ( cXML xml, const std::string& container = "" );
            ObjectNode& SaveToFile  ( const std::string& filePath, const std::string& container = "" );
            cXML                    SaveToXML   ( const std::string& container = "", int mode = 0 );

        protected:
            eShareLevel m_shareLevel;

            ObjectNode& m_sint;
    };

}

#endif // CSTORAGE_H
