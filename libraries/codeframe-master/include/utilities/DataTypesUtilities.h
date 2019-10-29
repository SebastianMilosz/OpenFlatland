#ifndef DataTypesUtilitiesH
#define DataTypesUtilitiesH

#define PACK_STRUCT_BEGIN
#define PACK_STRUCT_STRUCT __attribute__ ((__packed__))
#define PACK_STRUCT_END
#define PACK_STRUCT_FIELD(x) x __attribute__ ((aligned(4)))

#include <cstdlib>
#include <inttypes.h>

#include "MathUtilities.h"

#include "LoggerUtilities.h"

enum eOPR { SET = 0, GET = 1, CHANGE, INCREASE, DECREAS, DEFAULT };

// Typy Danych
typedef int8_t   bool_t;
typedef int8_t   s8_t;
typedef uint8_t  u8_t;
typedef int16_t  s16_t;
typedef uint16_t u16_t;
typedef int32_t  s32_t;
typedef uint32_t u32_t;

#define PIXEL32 int32_t

#define SET_RGB(r,g,b)	(((r) << 16) | ((g) << 8) | (b))		///< Convert to RGB
#define GET_A_VALUE(c)	((BYTE)(((c) & 0xFF000000) >> 24))		///< Alpha color component
#define GET_R_VALUE(c)	((BYTE)(((c) & 0x00FF0000) >> 16))		///< Red color component
#define GET_G_VALUE(c)	((BYTE)(((c) & 0x0000FF00) >> 8))		///< Green color component
#define GET_B_VALUE(c)	((BYTE)((c) & 0x000000FF))				///< Blue color component

namespace utilities
{
    namespace data
    {
        class DataStorage
        {
            public:
               virtual void Add( const std::string& key, const std::string& value ) = 0;
               virtual void Get( const std::string& key, std::string& value ) = 0;
        };

        template<uint32_t S, typename T>
        class ConstStack
        {
            public:
                ConstStack() :
                    m_count( 0U )
                {

                }

                T Peek( size_t id )
                {
                    if ( id < m_count )
                    {
                        return m_dataTable[ id ];
                    }
                }

                bool_t Push( T value )
                {
                    if ( !IsFull() )
                    {
                        m_dataTable[ m_count++ ] = value;
                        return true;
                    }
                    return false;
                }

                T Pop()
                {
                    if ( !IsEmpty() )
                    {
                        return  m_dataTable[ m_count-- ];
                    }
                    return m_dummyValue;
                }

                size_t Size() const
                {
                    return m_count;
                }

                bool_t IsEmpty()
                {
                    return (Size() == 0U);
                }

                bool_t IsFull() const
                {
                    return (Size() == S);
                }
            private:
                T m_dataTable[ S ];
                size_t m_count;
                T m_dummyValue;
        };

        template<uint32_t S, typename T>
        class CircularBuffer
        {
            public:
                CircularBuffer() :
                    m_head( 0U ),
                    m_tail( 0U ),
                    m_count( 0U ),
                    m_peekPos( 0U )
                {
                    LOGGER( LOG_INFO << "Constructor - m_head:" << m_head << "m_tail:" << m_tail << "m_count:" << m_count << "m_peekPos:" << m_peekPos);
                }

                T PeekNext()
                {
                    LOGGER( LOG_INFO << "PeekNext_prew - m_head:" << m_head << "m_tail:" << m_tail << "m_count:" << m_count << "m_peekPos:" << m_peekPos);

                    m_peekPos = (m_peekPos - 1U) % m_count;

                    size_t headtmp = (m_head - m_peekPos) % S;

                    LOGGER( LOG_INFO << "PeekNext      - m_head:" << m_head << "m_tail:" << m_tail << "m_count:" << m_count << "m_peekPos:" << m_peekPos << " headtmp: " << headtmp << " value: " << m_dataTable[ headtmp ]);

                    return m_dataTable[ headtmp ];
                }

                T PeekPrew()
                {
                    LOGGER( LOG_INFO << "PeekPrew_prew - m_head:" << m_head << "m_tail:" << m_tail << "m_count:" << m_count << "m_peekPos:" << m_peekPos);

                    m_peekPos = (m_peekPos + 1U) % (m_count+1);

                    size_t headtmp = (m_head - (m_peekPos-1)) % S;

                    LOGGER( LOG_INFO << "PeekPrew      - m_head:" << m_head << "m_tail:" << m_tail << "m_count:" << m_count << "m_peekPos:" << m_peekPos << " headtmp: " << headtmp << " value: " << m_dataTable[ headtmp ]);

                    return m_dataTable[ headtmp ];
                }

                void PeekReset()
                {
                    m_peekPos = 0U;
                }

                void Push( const T value )
                {
                    m_head = (m_head + 1U) % S;

                    m_dataTable[ m_head ] = value;
                    m_count++;

                    LOGGER( LOG_INFO << "Push - m_head:" << m_head << "m_tail:" << m_tail << "m_count:" << m_count << "m_peekPos:" << m_peekPos << " value: " << value );
                }

                T Pop()
                {
                    if ( m_count != 0U )
                    {
                        m_tail = (m_tail + 1U) % S;
                        m_count--;

                        LOGGER( LOG_INFO << "Pop - m_head:" << m_head << "m_tail:" << m_tail << "m_count:" << m_count << "m_peekPos:" << m_peekPos);

                        return  m_dataTable[ m_tail ];
                    }

                    LOGGER( LOG_INFO << "Pop - m_dummyValue");
                    return m_dummyValue;
                }

                size_t Size() const
                {
                    return m_count;
                }

                bool_t IsEmpty() const
                {
                    if ( m_count == 0U )
                    {
                        return true;
                    }

                    return false;
                }

                bool_t IsFull() const
                {
                    return (Size() == S);
                }

                void Save( DataStorage& ds ) const
                {
                    LOGGER( LOG_INFO << "Save" );
                    for ( size_t n = 0U; n < m_count; n++ )
                    {
                        ds.Add( std::string("ConsoleHistoryData") + utilities::math::IntToStr( n ), m_dataTable[ m_tail + n + 1U ] );
                    }

                    ds.Add( "ConsoleHistoryDataCount", utilities::math::IntToStr( m_count ) );
                }

                void Load( DataStorage& ds )
                {
                    LOGGER( LOG_INFO << "Load" );

                    std::string cntStr("");
                    ds.Get( "ConsoleHistoryDataCount", cntStr );
                    size_t cnt = utilities::math::StrToInt( cntStr );
                    for ( size_t n = 0U; n < cnt; n++ )
                    {
                        std::string valueStr("");
                        ds.Get( std::string("ConsoleHistoryData") + utilities::math::IntToStr( n ), valueStr );
                        Push( (T)valueStr );
                    }
                }

            private:
                T m_dataTable[ S ];
                size_t m_head;
                size_t m_tail;
                size_t m_count;
                size_t m_peekPos;
                T m_dummyValue;
        };

        enum ESpecialColor { CL_NONE = -1 };

        /*****************************************************************************/
        /**
          * @brief Struktura danych koloru
         **
        ******************************************************************************/
        template< class T >
        class Color
        {
            public:
                Color(const Color& cl)
                {
                    R = cl.R; G = cl.G; B = cl.B; A = cl.A; M = cl.M;
                }

                Color( T r = CL_NONE, T g = CL_NONE, T b = CL_NONE, T a = CL_NONE, T m = CL_NONE )
                {
                    R = r; G = g; B = b; A = a; M = m;
                }

                bool operator == ( const Color &cl ) const
                {
                    if(R == cl.R && G == cl.G && B == cl.B && A == cl.A && M == cl.M) return true;
                    else return false;
                }

                bool operator != ( const Color &cl ) const
                {
                    return !(*this == cl);
                }

                Color& operator=(const Color& c)
                {
                    if (this == &c)
                        return *this;
                    R = c.R; G = c.G; B = c.B; A = c.A; M = c.M;
                    return *this;
                }

                PIXEL32 GetPixel() const
                {
                    return (PIXEL32)((((int32_t)R) << 16) | (((int32_t)G) << 8) | ((int32_t)B));
                }

                bool IsValid() const
                {
                    if(R == CL_NONE || G == CL_NONE || B == CL_NONE) return false;
                    else return true;
                }

                T R, G, B, A, M;
        };

        typedef Color<int> iColor;

       enum Fill_mode { FILL_FULL, FILL_WIDTH, FILL_HEIGHT, FILL_BEST };

       template< class T>
       class Point
        {
            public:
            Point(T x = 0, T y = 0, T z = 0)
            {
                X = x; Y = y; Z = z;
            }

            T X;
            T Y;
            T Z;
        };

        typedef Point<int>       iPoint;

        template< class T>
        class tRec
        {
            public:
            tRec(T x = 0, T y = 0, T width = 0, T height = 0)
            {
                X = x; Y = y; W = width; H = height;
            }

            tRec( const tRec<int>& src )
            {
                X = (T)src.X; Y = (T)src.Y; W = (T)src.W; H = (T)src.H;
            }

            tRec( const tRec<float>& src )
            {
                X = (T)src.X; Y = (T)src.Y; W = (T)src.W; H = (T)src.H;
            }

            tRec( const tRec<double>& src )
            {
                X = (T)src.X; Y = (T)src.Y; W = (T)src.W; H = (T)src.H;
            }

            // Przeciazamy operator mnozenia
            tRec operator*( const float& c ) const
            { return tRec( (T)(X * c), (T)(Y * c), (T)(W * c), (T)(H * c) ); }

            tRec operator*( const tRec& c ) const
            { return tRec( (T)(X * (double)c.X), (T)(Y * (double)c.Y), (T)(W * (double)c.W), (T)(H * (double)c.H) ); }

            tRec operator/( const tRec& c ) const
            {
                tRec retRec;
                if(c.X != 0) retRec.X = (T)(X / (double)c.X);
                if(c.Y != 0) retRec.Y = (T)(Y / (double)c.Y);
                if(c.W != 0) retRec.W = (T)(W / (double)c.W);
                if(c.H != 0) retRec.H = (T)(H / (double)c.H);

                return retRec;
            }

            bool operator==(tRec &rhs)
            { if( X == rhs.X && Y == rhs.Y && W == rhs.W && H == rhs.H ) return true; else return false; }

            bool IsPointInside( int xp, int yp ) const
            {
                if( xp > X && yp > Y && xp < X + W && yp < Y + H ) return true;
                return false;
            }
            bool IsPointInside( Point<T>& pt ) const
            {
                if( pt.X > X && pt.Y > Y && pt.X < X + W && pt.Y < Y + H ) return true;
                return false;
            }

            T X;
            T Y;
            T W;
            T H;
        };

        typedef tRec<int>       iRec;
        typedef tRec<double>    dRec;

        iRec inline GetFitRec( const iRec& targetRec, const iRec& srcRec, int mode = FILL_BEST )
        {
            int width  = srcRec.W;
            int height = srcRec.H;

            switch( mode )
            {
                case FILL_FULL:
                {
                    if( targetRec.W  < srcRec.W  )
                        width  = targetRec.W;
                    if( targetRec.H < srcRec.H )
                        height = targetRec.H;
                    break;
                }
                case FILL_WIDTH:
                {
                    width  = targetRec.W;
                    height = (int)(height * (double)srcRec.W / srcRec.H);
                    break;
                }
                case FILL_HEIGHT:
                {
                    height = targetRec.H;
                    width = (int)(width * (double)srcRec.H / srcRec.W);
                    break;
                }
                case FILL_BEST:
                {
                    if((targetRec.W - srcRec.W) > (targetRec.H - srcRec.H) ) {
                        width  = targetRec.W;
                        height = (int)(height * (double)srcRec.W / srcRec.H);
                    }
                    else {
                        height = targetRec.H;
                        width = (int)(width * (double)srcRec.H / srcRec.W);
                    }
                    break;
                }
            }

            return iRec( 0, 0, width, height );
        };

        dRec inline GetFitRecFactor( const iRec& targetRec, const iRec& srcRec, int mode = FILL_BEST )
        {
            iRec tempRec = GetFitRec( targetRec, srcRec, mode );
            return dRec( tempRec.X, tempRec.Y, (double)tempRec.W / (double)srcRec.W, (double)tempRec.H / (double)srcRec.H );
        };
    };
};

#endif // DataTypesUtilitiesH
