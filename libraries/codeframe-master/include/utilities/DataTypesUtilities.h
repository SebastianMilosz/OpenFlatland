#ifndef DataTypesUtilitiesH
#define DataTypesUtilitiesH

#define PACK_STRUCT_BEGIN
#define PACK_STRUCT_STRUCT __attribute__ ((__packed__))
#define PACK_STRUCT_END
#define PACK_STRUCT_FIELD(x) x __attribute__ ((aligned(4)))

#include <cstdlib>
#include <inttypes.h>

enum eOPR { SET = 0, GET = 1, CHANGE, INCREASE, DECREAS, DEFAULT };

// Typy Danych
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
