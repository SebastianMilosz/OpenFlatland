#include "utilities/LoggerUtilities.h"

using namespace utilities::debug;
using namespace utilities::math;
using namespace utilities::file;
using namespace utilities::system;

cLog::cLogEntryContainer cLog::m_logContainer;
bool        cLog::MessageOnFatalError  = true;
bool        cLog::MessageOnError       = true;
bool        cLog::MessageOnWarning     = true;
bool        cLog::MessageOnInfo        = true;
bool        cLog::MessageOnACK         = true;

bool        cLog::ColectLogOnFatalError= false;
bool        cLog::ColectLogOnError     = false;
bool        cLog::ColectLogOnWarning   = false;
bool        cLog::ColectLogOnInfo      = false;
bool        cLog::ColectLogOnACK       = false;

std::string cLog::m_LogPath             = "";
int         cLog::m_LogDebugLevel       = 7;
int         cLog::m_AutoFlushMode       = 0;

WrMutex     cLog::m_mutex;

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog::cLogEntryContainer::cLogEntryContainer()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog::cLogEntryContainer::~cLogEntryContainer()
{
    for( unsigned int n = 0U; n < logEntrys.size(); n++ )
    {
        sLogEntry* temp = logEntrys.at( n );
        delete temp;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string cLog::GetLogHeader()
{
    std::string fileLine( "" );

            fileLine += "<html><head><style type=\"text/css\">";
            fileLine += " table { border:1px solid dodgerblue; border-collapse: collapse; width:100%; } ";
            fileLine += " td.hd { width:60px; color:black; background-color:gray; font-weight:bolder; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.ht { width:60px; color:black; background-color:gray; font-weight:bolder; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.hf { width:60px; color:black; background-color:gray; font-weight:bolder; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.hm { width:80px; color:black; background-color:gray; font-weight:bolder; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.hl { width:60px; color:black; background-color:gray; font-weight:bolder; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.h  { color:black; background-color:gray; font-weight:bolder; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.f  { color:red; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.e  { color:red; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.w  { color:black; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += " td.i  { color:blue; padding-left:5px; padding-right:5px; border:1px solid dodgerblue; } ";
            fileLine += "</style></head>";

    return fileLine;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string cLog::GetLogTableHeader()
{
    std::string fileLine( "" );

        fileLine += "<table>";
        fileLine += "<tr>";
        fileLine +=      "<td class=\"hd\"> Date </td>"
                         "<td class=\"ht\"> Time </td>"
                         "<td class=\"hf\"> IP </td>"
                         "<td class=\"hm\"> MAC </td>"
                         "<td class=\"hl\"> Port </td>"
                         "<td class=\"h\"> Info </td>";
        fileLine += "</tr>";

    return fileLine;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::SetLogDebugLevel( int debugLevel )
{
    m_LogDebugLevel = debugLevel;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::SetLogPath( const std::string& p )
{
    m_AutoFlushMode = 1;

    std::string filePath( p + std::string("/log.txt") );

    // Sprawdzamy czy plik istnieje jesli nie to tworzymy i dodajmey naglowek jesli trzeba
    if ( !IsFileExist(filePath) )
    {
        FILE *fp;
        std::string fileLine( "" );
        fp = fopen( filePath.c_str(), "a+" );

        fileLine += GetLogHeader();
        fileLine += GetLogTableHeader();

        fputs ( fileLine.c_str() , fp );
        fclose(fp);
    }

    m_LogPath = filePath;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog::cLogEntryContainer::sLogEntry::sLogEntry(int loglevel,
                                               const std::string& text,
                                               const std::string& datePoint,
                                               const std::string& timePoint,
                                               const std::string& filePoint,
                                               const std::string& methodPoint,
                                               int linePoint,
                                               const std::string& mac,
                                               const std::string& iP,
                                               const std::string& port)
{
    LogLevel    = loglevel;
    Text        = text;
    DatePoint   = datePoint;
    TimePoint   = timePoint;
    FilePoint   = filePoint;
    MethodPoint = methodPoint;
    LinePoint   = linePoint;
    Mac         = mac;
    IP          = iP;
    Port        = port;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::cLogEntryContainer::AddEntry( sLogEntry* le )
{
    logEntrys.push_back( le );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::cLogEntryContainer::AddEntry( int level,
                                        const std::string& text,
                                        const std::string& datePoint,
                                        const std::string& timePoint,
                                        const std::string& filePoint,
                                        const std::string& methodPoint,
                                        int linePoint,
                                        const std::string& mac,
                                        const std::string& iP,
                                        const std::string& port )
{
    AddEntry( new cLogEntryContainer::sLogEntry( level, text, datePoint, timePoint, filePoint, methodPoint, linePoint, mac, iP, port ) );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::cLogEntryContainer::SaveAsCSV( const std::string& path )
{
    FILE *fp;
    fp = fopen( path.c_str(), "a" );

    if ( fp != NULL )
    {
        for( unsigned int n = 0U; n < logEntrys.size(); n++ )
        {
            sLogEntry* temp = logEntrys.at( n );
            std::string logLine =   temp->DatePoint + ", " +
                                    temp->TimePoint + ", " +
                                    temp->FilePoint + ", " +
                                    temp->MethodPoint + ", " +
                                    IntToStr( temp->LinePoint ) + ", " +
                                    temp->Text + "\n";
            fputs ( logLine.c_str() ,fp );
        }

        fclose(fp);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::cLogEntryContainer::SaveAsHTML( const std::string& path )
{
    FILE *fp;
    std::string fileLine("");
    std::string fileExt("html");

    // Zmieniamy rozszerzenie na .html bo moglo zostac przekazane jakies inne
    std::string newPath = ChangeFileExtension( path, fileExt );

    // Teraz sprawdzamy czy istnieje jesli nie to musimy na poczatku dodac definicje stylow
    bool newFile = !IsFileExist( newPath );

    fp = fopen( newPath.c_str(), "a" );

    if ( fp != NULL )
    {
        if( newFile )
        {
            fileLine += GetLogHeader();
        }
        else
        {
            // Usuwamy z konca znacznik /html

        }

        fileLine += "<table>";
        fileLine += "<tr>";
        fileLine +=      "<td class=\"hd\"> Date </td>"
                         "<td class=\"ht\"> Time </td>"
                         "<td class=\"hf\"> IP </td>"
                         "<td class=\"hm\"> MAC </td>"
                         "<td class=\"hl\"> Port </td>"
                         "<td class=\"h\"> Info </td>";
        fileLine += "</tr>";

        for( unsigned int n = 0; n < logEntrys.size(); n++ )
        {
            std::string lineStyle("");

            fileLine += "<tr>";

            sLogEntry* temp = logEntrys.at( n );

            if( temp->LogLevel == LT_FATAL   ) lineStyle = "f";
            if( temp->LogLevel == LT_ERROR   ) lineStyle = "e";
            if( temp->LogLevel == LT_WARNING ) lineStyle = "w";
            if( temp->LogLevel == LT_INFO    ) lineStyle = "i";
            if( temp->LogLevel == LT_ACK     ) lineStyle = "a";

            fileLine    +=   "<td class=\"" + lineStyle + "\">" + temp->DatePoint + "</td>" +
                             "<td class=\"" + lineStyle + "\">" + temp->TimePoint + "</td>" +
                             "<td class=\"" + lineStyle + "\">" + temp->IP + "</td>" +
                             "<td class=\"" + lineStyle + "\">" + temp->Mac + "</td>" +
                             "<td class=\"" + lineStyle + "\">" + temp->Port + "</td>" +
                             "<td class=\"" + lineStyle + "\">" + temp->Text + "</td>";

            fileLine += "</tr>\n";
        }

        fileLine += "</table></br></br>";

        fputs ( fileLine.c_str() , fp );

        fclose(fp);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog::cLog()
{
    Create();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog::~cLog()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::Message( const std::string& title, const std::string& msg, int type, int debugLevel )
{
    (void)debugLevel;

    std::string timeStamp( GetNow() );

    if ( LogPath != "" )
    {
        // File Loggin
        std::fstream f;
        f.open( LogPath.c_str(), std::ios::out|std::ios::app );
        if ( !f.fail() )
        {
            f << timeStamp << " : " << title << " : " << msg << "\n";
            f.close();
        }
    }

    OnMessage.Emit( timeStamp, title, msg, type );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::Flush( const std::string& path, int mode )
{
    if(cLog::m_AutoFlushMode == 0)
    {
        if( mode == 0)
            cLog::m_logContainer.SaveAsCSV( path );
        else if( mode == 1 )
            cLog::m_logContainer.SaveAsHTML( path );
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::Create()
{
    m_mode          = cLog::LM_SILENCE;
    m_type          = cLog::LT_WARNING;
    m_activeCmd     = cLog::LC_NO;
    m_created       = false;

    LogPath         = "";
    m_timePoint     = "";
    m_filePoint     = "";
    m_methodPoint   = "";
    m_linePoint     = 0;
    m_DebugLevel    = 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cLog::Destroy()
{

}

/*****************************************************************************/
/**
  * @brief logger is singleton so it returns instance
 **
******************************************************************************/
cLog& cLog::GetInstance( const std::string& date,
                         const std::string& time,
                         const std::string& file,
                         const std::string& method,
                         int line )
{
    static cLog Instance;

    m_mutex.Lock();

    Instance.m_lineOpen     = true;
    Instance.m_datePoint    = date;
    Instance.m_timePoint    = time;
    Instance.m_filePoint    = utilities::file::GetFileName( file );
    Instance.m_methodPoint  = method;
    Instance.m_linePoint    = line;
    Instance.m_DebugLevel   = 0;

    m_mutex.Unlock();

    return Instance;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( const char *s )
{
    m_mutex.Lock();
         if(m_activeCmd == cLog::LC_IP)   { m_IP   = s; }
    else if(m_activeCmd == cLog::LC_MAC)  { m_Mac  = s; }
    else if(m_activeCmd == cLog::LC_PORT) { m_Port = s; }
    else
    {
        m_line += std::string(" ") + s;
    }
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( const std::string& s )
{
    m_mutex.Lock();
         if(m_activeCmd == cLog::LC_IP)   { m_IP   = s; }
    else if(m_activeCmd == cLog::LC_MAC)  { m_Mac  = s; }
    else if(m_activeCmd == cLog::LC_PORT) { m_Port = s; }
    else
    {
        m_line += " " + s;
    }
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( bool b )
{
    m_mutex.Lock();
    if( b )  m_line += " true ";
    else m_line += " false ";
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( char c )
{
    m_mutex.Lock();
    m_line += std::string(" ") + std::string( 1, c );
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( unsigned char c )
{
    m_mutex.Lock();
    m_line += std::string(" ") + std::string( 1, c );
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( unsigned short n )
{
    m_mutex.Lock();
    m_line += " " + IntToStr( n );
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( short n )
{
    m_mutex.Lock();
    m_line += " " + IntToStr( n );
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( unsigned int n )
{
    m_mutex.Lock();
    m_line += " " + IntToStr( n );
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( int n )
{
    m_mutex.Lock();
    if(m_activeCmd == cLog::LC_PORT) { m_Port = IntToStr(n); }
    else
    {
        m_line += " " + IntToStr( n );
    }
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( unsigned long n )
{
    m_mutex.Lock();
    if(m_activeCmd == cLog::LC_PORT) { m_Port = IntToStr(n); }
    else
    {
        m_line += " " + IntToStr( n );
    }
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( long n )
{
    m_mutex.Lock();
    if(m_activeCmd == cLog::LC_PORT) { m_Port = IntToStr(n); }
    else
    {
        m_line += " " + IntToStr( n );
    }
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( float f )
{
    m_mutex.Lock();
    m_line += " " + FloatToStr( f );
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( double d )
{
    m_mutex.Lock();
    m_line += " " + FloatToStr( d );
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( long double d )
{
    m_mutex.Lock();
    m_line += " " + FloatToStr( d );
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( ELog_Type type )
{
    m_mutex.Lock();
    m_type = type;
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cLog& cLog::operator << ( ELog_Mode mode )
{
    m_mutex.Lock();
    m_mode = mode;
    m_activeCmd = cLog::LC_NO;
    m_mutex.Unlock();
    return *this;
}

/*****************************************************************************/
/**
  * @brief Stream command implementation
 **
******************************************************************************/
cLog& cLog::operator << ( ELog_Cmd cmd )
{
    m_mutex.Lock();

    if( cmd == cLog::LC_END )
    {
        if( m_type == cLog::LT_FATAL )
        {
            if( MessageOnFatalError )
            {
                Message( "Fatal Error", m_line, m_type, m_DebugLevel );
            }
            if( ColectLogOnFatalError )
            {
                m_logContainer.AddEntry( cLog::LT_FATAL, m_line, m_datePoint, m_timePoint, m_filePoint, m_methodPoint, m_linePoint, m_Mac, m_IP, m_Port );
            }
        }
        else if( m_type == cLog::LT_ERROR )
        {
            if( MessageOnError )
            {
                Message( "Error      ", m_line, m_type, m_DebugLevel );
            }
            if( ColectLogOnError )
            {
                m_logContainer.AddEntry( cLog::LT_ERROR, m_line, m_datePoint, m_timePoint, m_filePoint, m_methodPoint, m_linePoint, m_Mac, m_IP, m_Port );
            }
        }
        else if( m_type == cLog::LT_WARNING )
        {
            if( MessageOnWarning )
            {
                Message( "Warning    ", m_line, m_type, m_DebugLevel );
            }
            if( ColectLogOnWarning )
            {
                m_logContainer.AddEntry( cLog::LT_WARNING, m_line, m_datePoint, m_timePoint, m_filePoint, m_methodPoint, m_linePoint, m_Mac, m_IP, m_Port );
            }
        }
        else if( m_type == cLog::LT_INFO )
        {
            if( MessageOnInfo )
            {
                Message( "Information", m_line, m_type, m_DebugLevel );
            }
            if( ColectLogOnInfo )
            {
                m_logContainer.AddEntry( cLog::LT_INFO, m_line, m_datePoint, m_timePoint, m_filePoint, m_methodPoint, m_linePoint, m_Mac, m_IP, m_Port );
            }
        }
        else if( m_type == cLog::LT_ACK )
        {
            if( MessageOnACK )
            {
                Message( "Confirmation", m_line, m_type, m_DebugLevel );
            }
            if( ColectLogOnACK )
            {
                m_logContainer.AddEntry( cLog::LT_ACK, m_line, m_datePoint, m_timePoint, m_filePoint, m_methodPoint, m_linePoint, m_Mac, m_IP, m_Port );
            }
        }
        else if( m_type == cLog::LT_XML )
        {
            if( MessageOnACK )
            {
                Message( "Confirmation", m_line, m_type, m_DebugLevel );
            }
            if( ColectLogOnACK )
            {
                m_logContainer.AddEntry( cLog::LT_XML, m_line, m_datePoint, m_timePoint, m_filePoint, m_methodPoint, m_linePoint, m_Mac, m_IP, m_Port );
            }
        }

        m_line = "";
    }
    else if( cmd == cLog::LC_SESION_START )
    {

    }
    else if( cmd == cLog::LC_SESION_END )
    {

    }

    m_activeCmd = cmd;

    m_mutex.Unlock();

    return *this;
}

/*****************************************************************************/
/**
  * @brief Stream log level implementation
 **
******************************************************************************/
cLog& cLog::operator << ( ELog_Level dl  )
{
    m_mutex.Lock();

    m_DebugLevel = dl;

    m_mutex.Unlock();

    return *this;
}
