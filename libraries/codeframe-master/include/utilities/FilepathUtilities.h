#ifndef FILEPATH_HPP
#define FILEPATH_HPP

#include <sstream>
#include <list>
#include <string>
#include <locale>
#include <cstdlib>
#include <stdio.h>

namespace utilities
{
    class filepath
    {
        public:
        static void CreateFolder( std::string path )
        {
            //if(!CreateDirectory(path.c_str() ,NULL))
            //{
            //   return;
            //}
        }

        private:
        class path_delimiters
        {
        public:
            static bool is_path_delim(char c)      { return (c == '/') || (c == '\\'); }
            static bool is_drive_delim(char c)     { return (c == ':'); }
            static bool is_scheme_delim(char c)    { return (c == ':'); }
            static bool is_port_delim(char c)      { return (c == ':'); }
            static bool is_query_delim(char c)     { return (c == '?'); }
            static bool is_extension_delim(char c) { return (c == '.'); }

            static char path_delim()      { return '/'; }
            static char drive_delim()     { return ':'; }
            static char scheme_delim()    { return ':'; }
            static char port_delim()      { return ':'; }
            static char query_delim()     { return '?'; }
            static char extension_delim() { return '.'; }

            static const char* unknown_scheme() { return "unknown"; }
        };

        public:
        filepath( std::string path, std::string absolutePath = "" )
        {
            m_path  = path;
            m_apath = absolutePath;

            // Usuwamy z absolutnej ostatni \ jesli jest taki
            if( path_delimiters::is_path_delim( m_apath[ m_apath.size() ] ) )
            {
                m_apath = m_apath.substr ( 0, m_apath.length()-1 );
            }

            // Usuwamy z pierwszej pozycji \ jesli istnieje
            if( path_delimiters::is_path_delim( m_apath[ 0 ] ) )
            {
                m_apath = m_apath.substr ( 1, m_apath.length() );
            }

            // Usuwamy z absolutnej ostatni \ jesli jest taki
            if( path_delimiters::is_path_delim( m_path[ m_path.size() ] ) )
            {
                m_path = m_path.substr ( 0, m_path.length()-1 );
            }

            // Usuwamy z pierwszej pozycji \ jesli istnieje
            if( path_delimiters::is_path_delim( m_path[ 0 ] ) )
            {
                m_path = m_path.substr ( 1, m_path.length() );
            }

            m_is_absoluteLocal = false;
            m_is_relativeLocal = false;
            m_is_URL = false;
            m_is_UsbCam = false;

            Parse( m_path.c_str() );
        }

        std::string GetAbsolute()
        {
            if( m_is_absoluteLocal ) { return m_path; }
            return (m_apath + std::string("\\") + m_path);
        }

        std::string GetURL()
        {
           return m_path;
        }

        bool isLocalFile()
        {
            if( m_is_absoluteLocal || m_is_relativeLocal ) { return true; }
            return false;
        }

        bool isURL()
        {
            return m_is_URL;
        }

        bool isUSB()
        {
            return m_is_UsbCam;
        }

        private:
            void Parse(const char* s)
            {
                std::locale loc;
                // check for Windows path first
                if( std::isalpha(s[0], loc ) &&
                        path_delimiters::is_drive_delim(s[1]) &&
                        path_delimiters::is_path_delim(s[2]))
                {
                    m_drive = s[0];
                    m_is_absoluteLocal = true;
                    s += 3;
                }
                else if(std::isalpha(s[0], loc ) &&
                        path_delimiters::is_path_delim(s[0]))
                {
                    m_is_relativeLocal = true;
                }
                else if( m_path.find("USBCAM") != std::string::npos )
                {
                    m_is_UsbCam = true;
                }
                else if( m_path.find("http") != std::string::npos )
                {
                    m_is_URL = true;
                }
            }

            bool        m_is_absoluteLocal;
            bool        m_is_relativeLocal;
            bool        m_is_URL;
            bool        m_is_UsbCam;
            std::string m_path;
            std::string m_apath;
            std::string m_drive;
    };
} // end namespace

#endif // FILEPATH_HPP
