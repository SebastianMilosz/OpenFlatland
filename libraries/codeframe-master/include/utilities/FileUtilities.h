#ifndef fileUtilitiesH
#define fileUtilitiesH

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <vector>
#include <limits.h>

#ifdef WIN32
    #include <windows.h>
    #include <direct.h>
    #define getcwd _getcwd // stupid MSFT "deprecation" warning
#else
    #include <unistd.h>
#endif

namespace utilities
{
	namespace file
	{
	    std::string GetFileName( const std::string& pathname );

		/*****************************************************************************/
		/**
		  * @brief Zwraca rozszerzenie pliku
		 **
		******************************************************************************/
		inline std::string GetFileExtension( const std::string& filePath )
		{
		    std::string ext = filePath.c_str();
		    size_t dotPos = ext.rfind(".");
		    if(dotPos != std::string::npos)
		    {
		        ext.erase(0, dotPos + 1);
		    }

		    return ext;
        }

		/*****************************************************************************/
		/**
		  * @brief
		 **
		******************************************************************************/
		inline std::string ChangeFileExtension( const std::string& filePath, const std::string& newExt )
		{
		    std::string retPath = filePath;
		    size_t dotPos = retPath.rfind(".");
		    if(dotPos != std::string::npos)
		    {
		        retPath.erase( dotPos, retPath.length() );
		    }

		    retPath += "." + newExt;

		    return retPath;
        }

		/*****************************************************************************/
		/**
		  * @brief Zwraca prawde jesli plik istnieje w przeciwntm razie falsz
		 **
		******************************************************************************/
		inline bool IsFileExist( const std::string& filePath )
		{
		    FILE* ftestexist;
		    bool ret = true;
		    ftestexist = fopen(filePath.c_str(),"rb");
		    if (ftestexist==NULL)
		        ret = false;
		    else
		        fclose(ftestexist);
		    return ret;
        }

        #ifdef WIN32
		/*****************************************************************************/
		/**
		  * @brief
		 **
		******************************************************************************/
        inline void wtoc(CHAR* Dest, const WCHAR* Source)
        {
            int i = 0;

            while(Source[i] != '\0')
            {
                Dest[i] = (CHAR)Source[i];
                ++i;
            }
        }

		/*****************************************************************************/
		/**
		  * @brief
		 **
		******************************************************************************/
        inline void ctow(WCHAR* Dest, const CHAR* Source)
        {
            int i = 0;

            while(Source[i] != '\0')
            {
                Dest[i] = (WCHAR)Source[i];
                ++i;
            }
        }
        #endif

		/*****************************************************************************/
		/**
		  * @brief Zwraca katalog aplikacji
		 **
		******************************************************************************/
        inline std::string GetExecutablePath()
        {
            char buffer[PATH_MAX];
            char *answer = getcwd(buffer, sizeof(buffer));
            std::string s_cwd;
            if (answer)
            {
                s_cwd = answer;
            }

            return s_cwd;
        }
    }

    namespace url
    {
        class cIP
        {
            public:
                cIP( void );

                cIP& FromString ( const std::string& str );
                cIP& FromInteger( uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4 );

                uint32_t    ToIntAdr() const;
                uint8_t*    ToIntAdrPtr();
                std::string ToString();
            private:
                uint8_t addr[4];
        };
    }
};

#endif
