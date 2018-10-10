#ifndef CSERIALIZABLESTORAGE_H
#define CSERIALIZABLESTORAGE_H

#include "cxml.hpp"

namespace codeframe
{
    class cSerializableInterface;

    /*****************************************************************************/
    /**
      * @brief This class add storage functionality to cInstanceManager
      * @author Sebastian Milosz
      * @version 1.0
      * @note cSetializable
     **
    ******************************************************************************/
    class cSerializableStorage
    {
        public:
            enum eShareLevel
            {
                ShareThis = 0,
                ShareFull
            };

                     cSerializableStorage( cSerializableInterface& sint );
            virtual ~cSerializableStorage();

            cSerializableInterface& ShareLevel  ( eShareLevel level = ShareFull );
            cSerializableInterface& LoadFromFile( const std::string& filePath, const std::string& container = "", bool createIfNotExist = false );
            cSerializableInterface& LoadFromXML ( cXML xml, const std::string& container = "" );
            cSerializableInterface& SaveToFile  ( const std::string& filePath, const std::string& container = "" );
            cXML                    SaveToXML   ( const std::string& container = "", int mode = 0 );

        protected:
            eShareLevel m_shareLevel;

            cSerializableInterface& m_sint;
    };

}

#endif // CSERIALIZABLESTORAGE_H
