#ifndef cLoggerH
#define cLoggerH

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <ThreadUtilities.h>
#include <MathUtilities.h>
#include <FileUtilities.h>
#include <SysUtilities.h>
#include <sigslot.h>

#define LOGGERINS() utilities::debug::cLog::GetInstance()
#define LOGGER(p) utilities::debug::cLog::GetInstance( __DATE__, __TIME__, __FILE__, __FUNCTION__, __LINE__ ) << p << utilities::debug::cLog::LC_END         ///< Zawsze dolanczany do wynikowego kodu
#define LOGGERFX(p) utilities::debug::cLog::GetInstance( __DATE__, __TIME__, __FILE__, __FUNCTION__, __LINE__ ) << p << utilities::debug::cLog::LC_END       ///< Wycinany calkowicie z kodu releace
#define FLUSH(p) utilities::debug::cLog::Flush( p, 1 );
#define LOG_FATAL utilities::debug::cLog::LT_FATAL
#define LOG_ERROR utilities::debug::cLog::LT_ERROR
#define LOG_WARNING utilities::debug::cLog::LT_WARNING
#define LOG_INFO utilities::debug::cLog::LT_INFO
#define LOG_ACK utilities::debug::cLog::LT_ACK
#define LOG_XML utilities::debug::cLog::LT_XML

#define LOG_LEVEL0 utilities::debug::cLog::LC_LEVEL0
#define LOG_LEVEL1 utilities::debug::cLog::LC_LEVEL1
#define LOG_LEVEL2 utilities::debug::cLog::LC_LEVEL2
#define LOG_LEVEL3 utilities::debug::cLog::LC_LEVEL3
#define LOG_LEVEL4 utilities::debug::cLog::LC_LEVEL4
#define LOG_LEVEL5 utilities::debug::cLog::LC_LEVEL5
#define LOG_LEVEL6 utilities::debug::cLog::LC_LEVEL6
#define LOG_LEVEL7 utilities::debug::cLog::LC_LEVEL7

#define LOGCMD_IP           utilities::debug::cLog::LC_IP
#define LOGCMD_MAC          utilities::debug::cLog::LC_MAC
#define LOGCMD_PORT         utilities::debug::cLog::LC_PORT
#define LOGCMD_SESION_START utilities::debug::cLog::LC_SESION_START
#define LOGCMD_SESION_END   utilities::debug::cLog::LC_SESION_END

#define LOG_FILENAME "log.html"
#define LOG_PATH(p) utilities::debug::cLog::GetInstance().SetLogPath(p)

using namespace sigslot;

namespace utilities
{
	namespace debug
	{
		/*****************************************************************************/
		/**
		  * @class cLogger
		  * @ingroup Debug
		  * @author Sebastian.Milosz
		  * @date 8.07.2009
		  * @brief
		  * @note Jak na razie klasa ta tylko obuduje standardowa obsluge loga wxWidgets
		  * Powinno dzialac to tak jak dziala teraz w aplikacji
		  * Ale Interfejs powinien byc nie zmienny szczegolnie definy final endu
		  * bo bede juz w calym projekcie kozystal z tej funkcjonalnosci
		  * SAMPLE
		  *
		  * LOGGER( << LOG_SESION_START << "SesionName" );
		  * LOGGER( << LOG_ERROR << "test" << x );
		  * LOGGER( << LOG_SESION_END << "SesionName" );
		  *
		 **
		******************************************************************************/
		class cLog
		{
		    public:
		        enum ELog_Type
		        {
		            LT_FATAL,	    ///< Fatal error This Kill Application
		            LT_ERROR,	    ///< Error This Kill current Task
		            LT_WARNING,	    ///< Warning This may cause som problems
                    LT_INFO,	    ///< Information
                    LT_ACK,         ///< ACK Confirmation
                    LT_XML          ///< Logowany tekst to XML
		        };

		        enum ELog_Mode
		        {
		            LM_SILENCE,        ///< Ciche wywolanie informacje leca tylko do loga plikowego
		            LM_MESSAGE         ///< Glosne wywolanie informacje londuja do loga i wyswietlane jest okineko informacyjne
		        };

		        enum ELog_Cmd
		        {
		            LC_NO,
		            LC_END,
		            LC_SESION_START,	///<
		            LC_SESION_END,	    ///<
		            LC_IP,
		            LC_MAC,
                    LC_PORT
		        };

                enum ELog_Level
                {
                    LC_LEVEL0 = 0,      ///< Priorytet dla loga (0 - najważniejszy)
                    LC_LEVEL1,
                    LC_LEVEL2,
                    LC_LEVEL3,
                    LC_LEVEL4,
                    LC_LEVEL5,
                    LC_LEVEL6,
                    LC_LEVEL7
                };

		        class cLogEntryContainer
		        {
		            public:
		                cLogEntryContainer();
		                ~cLogEntryContainer();

		                struct sLogEntry
		                {
                            sLogEntry(int loglevel, std::string text, std::string datePoint, std::string timePoint,
                                       std::string filePoint, std::string methodPoint, int linePoint, std::string Mac = "", std::string IP = "127.0.0.1", std::string Port = "");

		                    std::string     Text;
		                    std::string     DatePoint;
		                    std::string     TimePoint;
		                    std::string     FilePoint;
		                    std::string     MethodPoint;
		                    std::string     Mac;
		                    std::string     IP;
		                    std::string     Port;
		                    int             LinePoint;
                            int             LogLevel;
		                };

		                void AddEntry( sLogEntry* le );
		                void AddEntry( int level, std::string text, std::string datePoint, std::string timePoint,
		                               std::string filePoint, std::string methodPoint, int linePoint,
		                               std::string mac, std::string iP, std::string port );

		                void SaveAsCSV( std::string path );
		                void SaveAsHTML( std::string path );

		            private:
		                std::vector<sLogEntry*> logEntrys;
		        };

			    ~cLog();									// default destructor

			    void Create();								// creates logger
			    void Destroy();								// destroys logger

			    void SetLogPath(std::string p);
                void SetLogDebugLevel( int debugLevel );

			    signal5<std::string, std::string, std::string, std::string, int> OnMessage;

                            std::string LogPath;

			    static cLog& GetInstance( std::string date = "",
		                                  std::string time = "",
		                                  std::string file = "",
		                                  std::string method = "",
		                                  int line = 0 );	// logger is singleton so it returns instance

			    cLog& operator << ( const char *s );		// text					[const char *]
			    cLog& operator << ( std::string s );		// text					[string]

			    cLog& operator << ( bool b );			// logical value		[bool]

			    cLog& operator << ( char c );			// character			[char]
			    cLog& operator << ( unsigned char c );	// character			[char]

			    cLog& operator << ( unsigned short n );	// number				[unsigned short]
			    cLog& operator << ( short n );			// number				[short]

			    cLog& operator << ( unsigned int n );	// number				[unsigned int]
			    cLog& operator << ( int n );				// number				[int]

			    cLog& operator << ( unsigned long n );	// number				[unsigned long]
			    cLog& operator << ( long n );			// number				[long]

			    cLog& operator << ( float f );			// floating point		[float]
			    cLog& operator << ( double d );			// floating point		[double]
			    cLog& operator << ( long double d );		// floating point		[long double]

			    cLog& operator << ( ELog_Type type );	    // message type			[ELog_Type]
			    cLog& operator << ( ELog_Mode mode );	    // message mode			[ELog_Mode]
                cLog& operator << ( ELog_Cmd  cmd  );	    // message command	    [ELog_Cmd ]
                cLog& operator << ( ELog_Level dl  );

			    static void Flush( std::string path, int mode );

		        // Konfiguracja loggera
		        static bool MessageOnFatalError;
			    static bool MessageOnError;
		        static bool MessageOnWarning;
		        static bool MessageOnInfo;
                static bool MessageOnACK;

		        static bool ColectLogOnFatalError;
			    static bool ColectLogOnError;
		        static bool ColectLogOnWarning;
		        static bool ColectLogOnInfo;
                static bool ColectLogOnACK;

                static int  m_AutoFlushMode;

		    private:
		        cLog();
                void Message( std::string title, std::string msg, int type , int debugLevel );

                static std::string GetLogHeader();
                static std::string GetLogTableHeader();

		        int             m_mode;             ///<
		        int             m_type;
			    bool		    m_created;			///< is logger created
			    bool            m_lineOpen;         ///< otwarta Linia nowe dane dopisuja sie do niej
			    std::string	    m_line;				///< path to log file
			    std::ofstream   m_logFile;			///< log file

		        std::string     m_datePoint;
		        std::string     m_timePoint;
			    std::string     m_filePoint;
			    std::string     m_methodPoint;
			    int             m_linePoint;
			    ELog_Cmd        m_activeCmd;
                std::string     m_Mac;
                std::string     m_IP;
                std::string     m_Port;
                int             m_DebugLevel;

		        static cLogEntryContainer m_logContainer;

		        // Mutex
                static WrMutex m_mutex;

                static std::string  m_LogPath;
                static int          m_LogDebugLevel;
		};
    }
}

#endif
