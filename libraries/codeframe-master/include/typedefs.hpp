#ifndef TYPEDEFS_HPP_INCLUDED
#define TYPEDEFS_HPP_INCLUDED

// Typy Danych
typedef int8_t   bool_t;
typedef char     char_t;
typedef int8_t   s8_t;
typedef uint8_t  u8_t;
typedef int16_t  s16_t;
typedef uint16_t u16_t;
typedef int32_t  s32_t;
typedef uint32_t u32_t;
typedef float    float32_t;

#define PIXEL32 int32_t

#define SET_RGB(r,g,b)	(((r) << 16) | ((g) << 8) | (b))		///< Convert to RGB
#define GET_A_VALUE(c)	((BYTE)(((c) & 0xFF000000) >> 24))		///< Alpha color component
#define GET_R_VALUE(c)	((BYTE)(((c) & 0x00FF0000) >> 16))		///< Red color component
#define GET_G_VALUE(c)	((BYTE)(((c) & 0x0000FF00) >> 8))		///< Green color component
#define GET_B_VALUE(c)	((BYTE)((c) & 0x000000FF))				///< Blue color component

#endif // TYPEDEFS_HPP_INCLUDED
