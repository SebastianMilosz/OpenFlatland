#ifndef MJPEGVIDEOCAPTURE_H
#define MJPEGVIDEOCAPTURE_H

#include <sstream>
#include <Image.h>
#include <smartpointer.h>

#include "IVideoCapture.h"
#include "ActiveSocket.h"

class KMMJPEGVideoProvider;

class mjpegVideoCapture : public cIVideoCapture
{
    public:
        mjpegVideoCapture();
        mjpegVideoCapture(const std::string sourceId);
        virtual ~mjpegVideoCapture();
        virtual bool open();
        virtual bool setSource(int derviceId);
        virtual bool setSource(std::string sourceId);
        virtual bool isOpened() const;
        virtual void release();
        virtual bool read( cImage& image );
        virtual std::string toString();
    protected:
    private:
        std::string           m_sourceId;
        std::string           m_Host;
        std::string           m_Request;
        uint16_t              m_Port;
        KMMJPEGVideoProvider *m_MJPEGVProvider;
};

class KMMJPEGVideoProvider
{
    public:
        class KMUri
        {
            public:
                std::string QueryString;
                std::string Path;
                std::string Protocol;
                std::string Host;
                std::string Port;

                static KMUri Parse(const std::string &uri);
        };

    private:
        enum STATE_e
        {
            S_CONNECT,
            S_SEND_REQUEST,
            S_SEEK_HEADER,
            S_ANALIZE_HEADER,
            S_SEEK_BOUNDARY,
            S_ANALIZE_BOUNDARY,
            S_DOWNLOAD_JPEG,
            S_DISCONNECT
        };

        class HTTPHeader_c
        {
            public:
                uint8_t     VersionMajor;
                uint8_t     VersionMinor;
                uint16_t    ResultCode;
                uint32_t    ImageLength;
                char        ContentType[128];
                char        MJPEGBoundary[128];
                uint8_t     MJPEGBoundaryLength;
        };

        cImage        m_VideoFrame;

        std::string   m_Host;
        std::string   m_Request;
        uint16_t      m_Port;
        bool          m_Connected;
        CActiveSocket m_TCPClient;
        HTTPHeader_c  m_HTTPHeader;
        STATE_e       m_State;
        uint32_t      m_JPEGSize;

        const uint32_t  m_BufferSize;
        uint8_t        *m_Buffer;
        uint32_t        m_BufferTail;
        uint32_t        m_BufferHead;
        uint32_t        m_BufferPreviousTail;
        uint32_t        m_BufferPreviousHead;

        const uint32_t  m_SeekBufferSize;
        uint8_t        *m_SeekBuffer;
        uint8_t         m_SeekBufferTail;
        uint32_t        m_SeekLastResultIdx;
        uint32_t        m_SeekCounter;
        uint32_t        m_SeekLimit;

        void           Init();
        void           InitializeBuffer();
        void           InitializeSeekBuffer(const char *dataToFind,
                                            uint32_t    limit);
        bool           LoadBuffer(uint32_t maxBytes);
        int32_t        FindStringInBuffer();
        bool           AnalizeHeader(const uint8_t *header,
                                     uint32_t       headerSize);
        const uint8_t *FindCharData(const uint8_t *source,
                                    const uint8_t *lookFor,
                                    uint32_t       sourceSize);
        const uint8_t *FindCharData(const uint8_t *source,
                                    const char    *lookFor,
                                    uint32_t       sourceSize);
        uint32_t       MemcpyUntil(const uint8_t *source,
                                   uint8_t       *destination,
                                   const uint8_t *sourceEndPoint);
        bool           AnalizeBoundaryInfo(const uint8_t *temporaryBuffer,
                                           uint32_t       temporaryBufferSize);
    public:
        KMMJPEGVideoProvider(std::string &host,
                             std::string &request,
                             uint16_t      port = 80);
        ~KMMJPEGVideoProvider();

        bool Connect();
        bool ReadStream();
        cImage& GetImage();
};

#endif // MJPEGVIDEOCAPTURE_H
