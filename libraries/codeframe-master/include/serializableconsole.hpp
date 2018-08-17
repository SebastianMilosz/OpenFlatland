#ifndef _SERIALIZABLECONSOLE_H
#define _SERIALIZABLECONSOLE_H

#include <TextUtilities.h>

namespace codeframe
{

    class cSerializableConsole : public cSerializable
    {
        public:
            cSerializableConsole( std::string objName, std::istream& istr = std::cin, std::ostream& ostr = std::cout, std::ostream& estr = std::cerr ) :
            cSerializable( objName, NULL ),
            m_drawCommandPrompt(true),
            m_istr(istr), m_ostr(ostr), m_estr(estr)
            {

            }

            // Wyswietla panel informacyjny aplikacji
            virtual void OnCustomCommand(std::string cmd = "", std::string val = "")
            {

            }

            std::istream& ConsoleIn()      { return m_istr; }
            std::ostream& ConsoleOut()     { return m_ostr; }
            std::ostream& ConsoleLogOut()  { return m_ostr; }
            std::ostream& ConsoleErr()     { return m_estr; }

            void ExecuteCmd()
            {
                std::string cmd;
                std::string cmdUp;

                do
                {
                    if( m_drawCommandPrompt ) { ConsoleOut() << '>'; }

                    char cmdchartab[256];
                    ConsoleIn().getline(cmdchartab, 256);

                    if( cmdchartab[0] == 0x00 ) continue;

                    cmd = std::string( cmdchartab );

                    cmdUp = utilities::text::stringtoupper( cmd );

                    // Komedypierwotne sterujace
                         if( cmdUp.substr(0, 11) == "SCRIPT LINE" )    // Wykonanie jednej linii skryptu
                    {
                        std::string scriptString = cmd.substr(12);

                        this->LuaRunString( scriptString );
                    }
                    else if( cmdUp.substr(0, 11) == "SCRIPT FILE" )    // Wykonanie pliku skryptu
                    {
                        std::string scriptFile = cmd.substr(12);

                        this->LuaRunFile( scriptFile );
                    }
                    else
                    {
                        std::vector<std::string> tokens;

                        utilities::text::split(cmdUp, " ", tokens);

                        if(tokens.size() == 1) { OnCustomCommand(tokens[0]); }
                        else                   { OnCustomCommand(tokens[0], tokens[1]); }
                    }
                }
                while( cmdUp != "EXIT" );
            }

        protected:
            std::istream& m_istr;
            std::ostream& m_ostr;
            std::ostream& m_estr;

            bool m_drawCommandPrompt;
    };

}

#endif
