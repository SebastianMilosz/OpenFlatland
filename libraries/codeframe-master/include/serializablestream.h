#ifndef CSERIALIZABLESTREAM_H
#define CSERIALIZABLESTREAM_H

#include <serializable.h>
#include <MathUtilities.h>
#include <exception>
#include <stdexcept>
#include <ThreadUtilities.h>
#include <mjpegVideoCapture.h>

namespace codeframe
{

    class cSerializableStream : public cSerializable,  WrThread
    {
    public:
            std::string Role()      { return "Stream"; }
            std::string Class()     { return "cSerializableStream"; }
            std::string BuildType() { return "Dynamic"; }

    public:
                     cSerializableStream( std::string name, cSerializable* parentObject );
            virtual ~cSerializableStream();

            Property_Str Name;
            Property_Int State;
            Property_Int Mode;
            Property_Int Enable;
            Property_Str Address;
            Property_Int Port;

            bool TestStream();

            cImage& GetThumbnail();

            // Sygnaly zmiany
            signal1<cSerializableStream*> signalThumbnailChanged;
            signal1<cSerializableStream*> signalFrameChanged;

    private:
            bool StartProcessing();
            bool Processing();
            bool EndProcessing();

            void FrameCntInc();
            void FrameErrCntInc();
            void FPSFrameCntInc();
            int  GetFPS();

            void OnEnable  ( Property* prop );
            void OnMode    ( Property* prop );
            void OnAddress ( Property* prop );
            void OnPort    ( Property* prop );
            void OnUpdate1s( void );
            void OnVideoCaptureStateChanged(int val);

            WrTimer           m_Timer_1Sec;
            cImage            m_frame;
            cImage            m_frameThumbnail;
            mjpegVideoCapture m_MJPEGVideoCapture;
            int m_framecnt;
            int m_frameerrcnt;
            int m_framefpscnt;
            int m_fpscnt;

            bool m_force_thumbnail;
    };

}

#endif // CSERIALIZABLESTREAM_H
