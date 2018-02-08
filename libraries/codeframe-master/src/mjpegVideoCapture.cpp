#include "mjpegVideoCapture.h"

#include <algorithm>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
mjpegVideoCapture::mjpegVideoCapture()
{
    m_Opened         = false;
    m_MJPEGVProvider = NULL;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
mjpegVideoCapture::mjpegVideoCapture(const std::string sourceId)
{
    m_Opened         = false;
    m_MJPEGVProvider = NULL;
    setSource(sourceId);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
mjpegVideoCapture::~mjpegVideoCapture()
{
    if(m_MJPEGVProvider) delete m_MJPEGVProvider;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool mjpegVideoCapture::setSource(int derviceId)
{
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool mjpegVideoCapture::setSource(std::string sourceId)
{
    m_sourceId = sourceId;
    KMMJPEGVideoProvider::KMUri l_Uri = KMMJPEGVideoProvider::KMUri::Parse(sourceId);
    m_Host = l_Uri.Host;
    if(l_Uri.Port != "") std::istringstream (l_Uri.Port) >> m_Port;
    else m_Port = 80;
    m_Request = l_Uri.Path + l_Uri.QueryString;
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool mjpegVideoCapture::open()
{
    if(m_MJPEGVProvider) delete m_MJPEGVProvider;

    std::string l_Request = "GET " +
                            m_Request +
                            " HTTP/1.0\r\n"
                            "\r\n";

    m_MJPEGVProvider = new KMMJPEGVideoProvider(m_Host,
                                                l_Request,
                                                m_Port);
    m_Opened = m_MJPEGVProvider->Connect();
    return m_Opened;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool mjpegVideoCapture::isOpened() const
{
    return m_Opened;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void mjpegVideoCapture::release()
{
    if(m_MJPEGVProvider) delete m_MJPEGVProvider;
    m_MJPEGVProvider = NULL;
    m_Opened = 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool mjpegVideoCapture::read( cImage& image )
{
    if(m_MJPEGVProvider->ReadStream())
    {
        cImage& img = m_MJPEGVProvider->GetImage();

        img.Move_To( image );
        return true;
    }

    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string mjpegVideoCapture::toString()
{
    return std::string("MJPEG Camera");
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
KMMJPEGVideoProvider::KMMJPEGVideoProvider(std::string &host,
                                           std::string &request,
                                           uint16_t      port)
:   m_Connected(false),
    m_Buffer(0),
    m_BufferSize(1024 * 1024),
    m_SeekBuffer(0),
    m_SeekBufferSize(16),
    m_Host(host),
    m_Request(request),
    m_Port(port)
{
    m_Buffer = new uint8_t[m_BufferSize];
    m_SeekBuffer = new uint8_t[m_SeekBufferSize];
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
KMMJPEGVideoProvider::~KMMJPEGVideoProvider()
{
    if(m_Buffer) delete [] m_Buffer;
    if(m_SeekBuffer) delete [] m_SeekBuffer;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void KMMJPEGVideoProvider::Init()
{
    m_State                 = S_CONNECT;
    m_BufferTail            = 0;
    m_BufferHead            = 0;
    m_BufferPreviousTail    = 0;
    m_BufferPreviousHead    = 0;
    m_SeekBufferTail        = 0;
    m_SeekLastResultIdx     = 0;
    m_SeekCounter           = 0;
    m_SeekLimit             = 1024;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void KMMJPEGVideoProvider::InitializeBuffer()
{
    m_BufferTail = 0;
    m_BufferHead = 0;
    m_BufferPreviousTail = 0;
    m_BufferPreviousHead = 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void KMMJPEGVideoProvider::InitializeSeekBuffer(const char *dataToFind,
                                             uint32_t limit)
{
    strcpy(reinterpret_cast<char*>(m_SeekBuffer), dataToFind);
    m_SeekBufferTail    = 0;
    m_SeekCounter       = 0;
    m_SeekLimit         = limit;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int32_t KMMJPEGVideoProvider::FindStringInBuffer()
{
    while(m_BufferTail < m_BufferHead)
    {
        if(m_SeekCounter < m_SeekLimit)
        {
            m_SeekCounter++;

            if(m_Buffer[m_BufferTail++] == m_SeekBuffer[m_SeekBufferTail])
            {
                m_SeekBufferTail++;
                if(m_SeekBuffer[m_SeekBufferTail] == 0)
                {
                    return m_BufferTail;
                }
            }
            else m_SeekBufferTail = 0;
        }
        else
        {
            m_State = S_DISCONNECT;
            return -1;
        }
    }

    return -1;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool KMMJPEGVideoProvider::LoadBuffer(uint32_t maxBytes)
{
    bool l_Result = false;
    uint32_t l_UnreadBytes = m_BufferHead - m_BufferTail;

    if(l_UnreadBytes < maxBytes)
    {
        uint32_t l_MaxBytesToRead = maxBytes - l_UnreadBytes;

        m_TCPClient.Receive(l_MaxBytesToRead);
        CActiveSocket::CSocketError l_SocketError = m_TCPClient.GetSocketError();
        if(l_SocketError == CActiveSocket::SocketSuccess)
        {
            uint32_t l_ReceivedBytes = m_TCPClient.GetBytesReceived();
            if(l_ReceivedBytes)
            {
                uint8_t *l_DataPointer = m_TCPClient.GetData();

                if(m_BufferHead + l_ReceivedBytes < m_BufferSize)
                {
                    memcpy(reinterpret_cast<void*>(&m_Buffer[m_BufferHead]),
                           reinterpret_cast<const void*>(l_DataPointer),
                           l_ReceivedBytes);
                    m_BufferHead += l_ReceivedBytes;
                    l_Result = true;
                }
            }
            else m_State = S_DISCONNECT;
        }
        else m_State = S_DISCONNECT;
    }
    else l_Result = true;

    return l_Result;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool KMMJPEGVideoProvider::AnalizeHeader(const uint8_t *header,
                                         uint32_t headerSize)
{
    const uint8_t *bByte_p;
    uint8_t sscanfResult;

    bByte_p = FindCharData(header, "HTTP/", headerSize);
    if(bByte_p == 0)
    {
        return false;
        //throw std::runtime_error("Ups! Probalby it isn't HTTP protocol");
    }

    sscanfResult = sscanf(reinterpret_cast<const char *>(header),
                          "HTTP/%d.%d %3d",
                          &m_HTTPHeader.VersionMajor,
                          &m_HTTPHeader.VersionMinor,
                          &m_HTTPHeader.ResultCode);

    if(sscanfResult != 3)
    {
        return false;
        //throw std::runtime_error("Ups! Can't recognize HTTP protocol version or HTTP result code");
    }

    if(m_HTTPHeader.ResultCode != 200)
    {
        return false;
        //std::stringstream ss;
        //ss << HTTP_HEADER.resultCode;
        //throw std::runtime_error("HTTP result code: " + ss.str());
    }

    bByte_p = FindCharData(header, "Content-Type: ", headerSize);
    if(bByte_p == 0)
    {
        return false;
        //throw std::runtime_error("Ups! Can't recognize HTTP header Content-Type");
    }


    bByte_p = FindCharData(bByte_p,
                           "multipart/x-mixed-replace",
                           sizeof("multipart/x-mixed-replace") - 1);
    if(bByte_p == 0)
    {
        return false;
        //throw std::runtime_error("Ups! HTTP Content-Type different from multipart/x-mixed-replace");
    }


    bByte_p = FindCharData(bByte_p,
                           "boundary=",
                           (header + headerSize) - bByte_p);
    if(bByte_p == 0)
    {
        return false;
        //throw std::runtime_error("Ups! Can't find boundary string");
    }

    const uint8_t *eChar_p;
    eChar_p = FindCharData(bByte_p,
                           "\r\n",
                           (header + headerSize) - bByte_p);

    uint32_t bytesCopied = MemcpyUntil(bByte_p,
                                       reinterpret_cast<uint8_t *>(&m_HTTPHeader.MJPEGBoundary),
                                       eChar_p - (sizeof("\r\n") - 1));

    m_HTTPHeader.MJPEGBoundaryLength = bytesCopied;
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const uint8_t *KMMJPEGVideoProvider::FindCharData(const uint8_t *source,
                                                  const uint8_t *lookFor,
                                                  uint32_t sourceSize)
{
    uint32_t sourceIdx = 0;
    uint32_t lookForIdx = 0;

    while(sourceIdx < sourceSize)
    {
        if(source[sourceIdx] == lookFor[lookForIdx])
        {
            lookForIdx++;
        }
        else
        {
            lookForIdx = 0;
        }

        sourceIdx++;

        if(lookFor[lookForIdx] == 0)
        {
            return const_cast<uint8_t *>(&source[sourceIdx]);
        }
    }

    return 0;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
const uint8_t *KMMJPEGVideoProvider::FindCharData(const uint8_t *source,
                                                  const char *lookFor,
                                                  uint32_t sourceSize)
{
    return FindCharData(source,
                        reinterpret_cast<const uint8_t *>(lookFor),
                        sourceSize);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
uint32_t KMMJPEGVideoProvider::MemcpyUntil(const uint8_t *source,
                                           uint8_t *destination,
                                           const uint8_t *sourceEndPoint)
{
    uint32_t bytesCopied = 0;

    while(source != sourceEndPoint)
    {
        *(destination++) = *(source++);
        bytesCopied++;
    }

    return bytesCopied;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool KMMJPEGVideoProvider::AnalizeBoundaryInfo(const uint8_t *temporaryBuffer,
                                               uint32_t temporaryBufferSize)
{
    const uint8_t *bByte_p;
    uint8_t        sscanfResult;

    bByte_p = FindCharData(temporaryBuffer,
                           "Content-Type: ",
                           temporaryBufferSize);
    if(bByte_p == 0)
    {
        return false;
        //throw std::runtime_error("Ups! Can't find image Content-Type");
    }

    bByte_p = FindCharData(bByte_p,
                           "image/jpeg",
                           sizeof("image/jpeg") - 1);
    if(bByte_p == 0)
    {
        return false;
        //throw std::runtime_error("Ups! Image Content-Type different from image/jpeg");
    }

    bByte_p = FindCharData(temporaryBuffer,
                           "Content-Length: ",
                           temporaryBufferSize);
    if(bByte_p == 0)
    {
        return false;
        //throw std::runtime_error("Ups! Can't find image Content-Length");
    }

    sscanfResult = sscanf(reinterpret_cast<const char *>(bByte_p),
                          "%d",
                          &m_HTTPHeader.ImageLength);
    if(sscanfResult != 1)
    {
        return false;
        //throw std::runtime_error("Ups! Can't recognize image Content-Length");
    }

    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool KMMJPEGVideoProvider::Connect()
{
    Init();
    InitializeBuffer();
    m_TCPClient.Initialize();
    m_TCPClient.SetReceiveTimeout(5000, 0);

    m_Connected = m_TCPClient.Open(reinterpret_cast<const uint8_t*>(m_Host.c_str()),
                                   m_Port);
    if(m_Connected) m_State = S_SEND_REQUEST;
    return m_Connected;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool KMMJPEGVideoProvider::ReadStream()
{
    while(m_State != S_DISCONNECT)
    {
        if(m_State == S_CONNECT)
        {
            Init();
            InitializeBuffer();
            m_TCPClient.Initialize();
            m_TCPClient.SetReceiveTimeout(5000, 0);

            m_Connected = m_TCPClient.Open(reinterpret_cast<const uint8_t*>(m_Host.c_str()),
                                           m_Port);
            if(m_Connected) m_State = S_SEND_REQUEST;
            else return false;
        }

        if(m_State == S_SEND_REQUEST)
        {
            if(m_TCPClient.Send(reinterpret_cast<const uint8_t*>(m_Request.c_str()),
                                m_Request.size()))
            {
                InitializeSeekBuffer("\r\n\r\n", 8 * 1024);
                m_State = S_SEEK_HEADER;
            }
            else m_State = S_DISCONNECT;
        }

        if(m_State == S_SEEK_HEADER)
        {
            if(LoadBuffer(1024))
            {
                int32_t l_Result = FindStringInBuffer();
                if(l_Result > 0)
                {
                    m_BufferPreviousHead = l_Result;
                    m_State = S_ANALIZE_HEADER;
                }
            }
        }

        if(m_State == S_ANALIZE_HEADER)
        {
            if(AnalizeHeader(m_Buffer, m_BufferPreviousHead))
            {
                m_State = S_SEEK_BOUNDARY;
                InitializeSeekBuffer("\r\n\r\n", 1 * 1024);
            }
            else m_State = S_DISCONNECT;
        }

        if(m_State == S_SEEK_BOUNDARY)
        {
            if(LoadBuffer(256))
            {
                int32_t l_Result = FindStringInBuffer();
                if(l_Result > 0)
                {
                    m_BufferPreviousTail = m_BufferPreviousHead;
                    m_BufferPreviousHead = l_Result;
                    m_State = S_ANALIZE_BOUNDARY;
                }
            }
        }

        if(m_State == S_ANALIZE_BOUNDARY)
        {
            if(AnalizeBoundaryInfo(&m_Buffer[m_BufferPreviousTail],
                                    m_BufferPreviousHead))
            {
                m_JPEGSize = 0;
                m_State = S_DOWNLOAD_JPEG;
            }
            else m_State = S_DISCONNECT;
        }

        if(m_State == S_DOWNLOAD_JPEG)
        {
            uint32_t l_BytesToDownload = m_HTTPHeader.ImageLength - m_JPEGSize;
            if(l_BytesToDownload > 1024) l_BytesToDownload = 1024;

            if(LoadBuffer(l_BytesToDownload))
            {
                m_JPEGSize = m_BufferHead - m_BufferPreviousHead;
                m_BufferTail = m_BufferHead;
                if(m_JPEGSize >= m_HTTPHeader.ImageLength)
                {
                    //FILE * pFile;
                    //pFile = fopen ("myfile.jpg" , "wb");
                    //fwrite (&m_Buffer[m_BufferPreviousHead] , 1 , m_JPEGSize, pFile);
                    //fclose (pFile);

                    m_VideoFrame.ReadJpegFromStream(&m_Buffer[m_BufferPreviousHead],
                                                    m_JPEGSize);

                    InitializeBuffer();
                    InitializeSeekBuffer("\r\n\r\n", 1 * 1024);
                    m_State = S_SEEK_BOUNDARY;
                    return true;
                }
            }
        }

        if(m_State == S_DISCONNECT)
        {
            m_TCPClient.Close();
            m_State = S_CONNECT;
            return false;
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cImage& KMMJPEGVideoProvider::GetImage()
{
    return m_VideoFrame;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
KMMJPEGVideoProvider::KMUri KMMJPEGVideoProvider::KMUri::Parse(const std::string &uri)
{
    KMUri result;

    typedef std::string::const_iterator iterator_t;

    if (uri.length() == 0)
        return result;

    iterator_t uriEnd = uri.end();

    iterator_t queryStart = std::find(uri.begin(), uriEnd, '?');

    iterator_t protocolStart = uri.begin();
    iterator_t protocolEnd = std::find(protocolStart, uriEnd, ':');

    if (protocolEnd != uriEnd)
    {
        std::string prot = &*(protocolEnd);
        if ((prot.length() > 3) && (prot.substr(0, 3) == "://"))
        {
            result.Protocol = std::string(protocolStart, protocolEnd);
            protocolEnd += 3;
        }
        else
            protocolEnd = uri.begin();
    }
    else
        protocolEnd = uri.begin();

    iterator_t hostStart = protocolEnd;
    iterator_t pathStart = std::find(hostStart, uriEnd, '/');

    iterator_t hostEnd = std::find(protocolEnd,
        (pathStart != uriEnd) ? pathStart : queryStart,
        ':');

    result.Host = std::string(hostStart, hostEnd);

    if ((hostEnd != uriEnd) && ((&*(hostEnd))[0] == ':'))
    {
        hostEnd++;
        iterator_t portEnd = (pathStart != uriEnd) ? pathStart : queryStart;
        result.Port = std::string(hostEnd, portEnd);
    }

    if (pathStart != uriEnd)
        result.Path = std::string(pathStart, queryStart);

    if (queryStart != uriEnd)
        result.QueryString = std::string(queryStart, uri.end());

    return result;
}
