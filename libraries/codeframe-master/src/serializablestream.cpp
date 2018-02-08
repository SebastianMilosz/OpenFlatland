#include <serializablestream.h>
#include <mjpegVideoCapture.h>
#include <FileUtilities.h>
#include <FilepathUtilities.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableStream::cSerializableStream( std::string name, cSerializable* parentObject ) : 
cSerializable( name,  parentObject ), WrThread(), m_Timer_1Sec(1000),
Name   (this, "NAME",    (std::string)"stream"   , cPropertyInfo().Kind(KIND_TEXT  ).Enable(true ).Description("Stream Nane")),
State  (this, "State",   (int)0			         , cPropertyInfo().Kind(KIND_ENUM  ).Enable(false).Description("Stream State").Enum("Created, Initialized, Connecting, Streaming, Error")),
Mode   (this, "MODE",    (int)0			         , cPropertyInfo().Kind(KIND_ENUM  ).Enable(true ).Description("Stream Mode").Enum("URL MJPG, Alice Server")),
Enable (this, "ENABLE",  (int)0                  , cPropertyInfo().Kind(KIND_LOGIC ).Enable(true ).Description("Enable Processing")),
Address(this, "ADDRESS", (std::string)"localhost", cPropertyInfo().Kind(KIND_URL   ).Enable(true ).Description("Enable Processing")),
Port   (this, "PORT",    (int)0                  , cPropertyInfo().Kind(KIND_NUMBER).Enable(true ).Description("Enable Processing"))
{
	SetPriority( WRTHREAD_MIN_PRIORITY );	
	
	Mode.signalChanged.connect        (this, &cSerializableStream::OnMode    );	
	Enable.signalChanged.connect      (this, &cSerializableStream::OnEnable  );
    Address.signalChanged.connect     (this, &cSerializableStream::OnAddress );
    Port.signalChanged.connect        (this, &cSerializableStream::OnPort    );
    m_Timer_1Sec.signalTimeout.connect(this, &cSerializableStream::OnUpdate1s);
    m_MJPEGVideoCapture.signalStateChanged.connect(this, &cSerializableStream::OnVideoCaptureStateChanged);
    
    LOGGER( LOG_INFO  << "Serializable Stream Create" );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cSerializableStream::~cSerializableStream()
{
    disconnect_all();
    
    LOGGER( LOG_INFO  << "Serializable Stream Delete" );
    
	TerminateProcessing();	
}	

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStream::OnPort( Property* prop )
{
    
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStream::OnUpdate1s( void )
{
    m_fpscnt = m_framefpscnt;
    m_framefpscnt = 0;
    m_force_thumbnail = true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStream::OnVideoCaptureStateChanged(int val)
{
    State = val;
}

/*****************************************************************************/
/**
  * @brief Reakcja na zmiane adresu url, rozlaczenie poprzedniego 
  * polaczenia i ustanowienie nowego
 **
******************************************************************************/
void cSerializableStream::OnAddress( Property* prop )
{
    IVideoCapture* l_IVideoCapture;
    l_IVideoCapture = reinterpret_cast<IVideoCapture*>( &m_MJPEGVideoCapture );

    if( l_IVideoCapture != NULL )
    {
        l_IVideoCapture->release();
        l_IVideoCapture->setSource( (std::string)*prop );
        l_IVideoCapture->open();
    } 

    LOGGER( LOG_INFO << "Parameter: " << prop->Name() << "Changed from: " << prop->PreviousValueString() << " To: " << prop->CurentValueString() );   
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStream::OnEnable( Property* prop )
{
    if( (bool)(*prop) == true ) { Start(); }
    else { Pause(); }	
    
    LOGGER( LOG_INFO << "Parameter: " << prop->Name() << "Changed from: " << prop->PreviousValueString() << " To: " << prop->CurentValueString() ); 
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStream::OnMode( Property* prop )
{
	
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableStream::TestStream()
{
    utilities::filepath fp( (std::string)Address, utilities::file::GetExecutablePath() );

	return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cImage& cSerializableStream::GetThumbnail()
{
    return m_frameThumbnail;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableStream::StartProcessing()
{
    LOGGER( LOG_INFO  << "Serializable Stream Thread Start" );

    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableStream::Processing()
{
    if( m_MJPEGVideoCapture.read( m_frame ) )
    {
        // Wykonanie miniaturki potrzebnej do reprezentacji tego strumienia
        if( m_force_thumbnail )
        {
            m_force_thumbnail = 0;
            
            m_frameThumbnail.FromImage( m_frame );
            
            m_frameThumbnail.Resize( 100, 80 );
            
            signalThumbnailChanged.Emit( this );
        }
    
        signalFrameChanged.Emit( this ) ;
        
        FPSFrameCntInc();
        Sleep( 25 );
    }
    
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool cSerializableStream::EndProcessing()
{
    LOGGER( LOG_INFO  << "Serializable Stream Thread End" );
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStream::FrameCntInc()
{
    m_framecnt++;

    if( m_framecnt > 9999 ) m_framecnt = 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStream::FrameErrCntInc()
{
    m_frameerrcnt++;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cSerializableStream::FPSFrameCntInc()
{
    m_framefpscnt++;

    FrameCntInc();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int cSerializableStream::GetFPS()
{
    return m_fpscnt;
}