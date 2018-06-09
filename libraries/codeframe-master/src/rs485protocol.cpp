#include "rs485protocol.h"
#include "crc.h"

#include <assert.h>
#include <LoggerUtilities.h>
#include <MathUtilities.h>

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerialProtocol::cSerialProtocol( CommunicationInterface* comdrv, int id, uint8_t addr ) :
        m_thisAdr( addr ),
        m_comdrv( comdrv ),
        c_HEADER_SIZE( 4 ),
        m_id( id ),
        dataSize( 0 ),
        m_STAT_DataSizeError( 0 ),
        m_STAT_CommunicationError( 0 ),
        m_STAT_CommunicationSucces( 0 )
    {
        if( m_comdrv == NULL )
        {
            LOGGER( LOG_ERROR << "NULL Device" );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerialProtocol::~cSerialProtocol()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cSerialProtocol::IsValid() const
    {
        if( m_comdrv == NULL )
        {
            return false;
        }
        return true;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cSerialProtocol::Synchronize( Property* prop, bool forceRead, bool forceWrite )
    {
        if( m_comdrv == NULL  ) { return false; }
        if( prop     == NULL  ) { return false; }

        cRegister& reg = prop->Info().GetRegister();

        if( reg.IsEnable() )
        {
            bool isError = false;
            int  regmode = reg.Mode();

            // Jesli nie pobieramy zmienna z urzadzenia
            if( (regmode & (int)R) && (prop->IsChanged() == false || forceRead == true) && forceWrite == false )
            {
                dataTable[ REG_MODE  ] = 0x0000 | m_thisAdr;    // Mode Read
                dataTable[ REG_START ] = reg.ReadRegister();    // Register Start
                dataTable[ REG_CNT   ] = 0x0001;                // Register Cnt

                if( m_comdrv->Write( m_id, (char*)dataTable, HEADER_8ByteSize) == HEADER_8ByteSize )
                {
                    if( m_comdrv->Read( m_id, (char*)dataTable, 8 ) == 8 )
                    {
                        if( dataTable[REG_START] == reg.ReadRegister() )
                        {
                            *prop = dataTable[ REG_DATA0 ];
                        }
                        else
                        {
                            isError = true;
                        }
                    }
                    else
                    {
                        isError = true;
                    }
                }
                else
                {
                    isError = true;
                }

                prop->CommitChanges();
            }
            // Jesli zmienna zostala zmieniona aktualizujemy do urzadzenia
            else if( ( regmode & (int)W ) && ( prop->IsChanged() == true || forceWrite == true ) )
            {
                dataTable[ REG_MODE  ] = 0x8000 | m_thisAdr;   // Mode Write
                dataTable[ REG_START ] = reg.WriteRegister();  // Register Start
                dataTable[ REG_CNT   ] = 0x0001;               // Register Cnt
                dataTable[ REG_DATA0 ] = (uint16_t)(*prop);    // Value

                if( m_comdrv->Write( m_id, (char*)dataTable, 8) == 8 )
                {
                    if( m_comdrv->Read( m_id, (char*)dataTable, 2 ) == 2 )
                    {
                        if( dataTable[0] != 0xAB00 )
                        {
                            isError = true;
                        }
                    }
                }

                prop->CommitChanges();
            }

            if( isError )
            {
                m_STAT_CommunicationError++;
                return false;
            }
            else
            {
                m_STAT_CommunicationSucces++;
            }
        }

        return true;
    }

    /*****************************************************************************/
    /**
      * @brief
      * @note Pierwsze 16b to calkowity rozmiar danych czyli: (2 + timestampDataSize + data.size)
      * Drugie 16b to rozmiar tablicy stepli czasowych
      * Następnie 4x16b x rozmiar tablicy stepli
      * A potem N bajtow danych
     **
    ******************************************************************************/
    bool cSerialProtocol::SynchronizeRange( Property* prop, uint16_t* data, uint16_t dataSize, uint64_t* timestampTable, uint16_t* timeStampTableSize64 )
    {
        bool isError = false;

        assert( m_comdrv != NULL );

        // Kolejne zapytanie o caly bufor przeslany jednym ciagiem
        if( dataSize > 0 && dataSize < USART_RX_BUFFER_DATA_SIZE )
        {
            cRegister& reg = prop->Info().GetRegister();

            dataTable[ REG_MODE  ] = 0x0000 | m_thisAdr;  // Mode Read Data
            dataTable[ REG_START ] = reg.ReadRegister();  // Register Start
            dataTable[ REG_CNT   ] = dataSize;            // Register Cnt

            if( m_comdrv->Write( m_id, (char*)dataTable, HEADER_8ByteSize) == HEADER_8ByteSize )
            {
                uint16_t readSize = HEADER_8ByteSize + 2 * dataSize;
                memset(dataTable, 0x00, sizeof(dataTable));

                uint16_t readSizeReal = m_comdrv->Read( m_id, (char*)dataTable, readSize );
                if( readSizeReal == readSize )
                {
                    if( dataTable[ REG_START ] == reg.ReadRegister() )
                    {
                        if( dataTable[ REG_CNT ] == dataSize )
                        {
                            unsigned int data_cnt = HEADER_16ByteSize;

                            // Pierwsze 16b to rozmiar tablicy probek czasowych
                            uint16_t     dataTimeStampTableSize   = dataTable[ data_cnt++ ];
                            unsigned int dataTimeStampTableSize64 = dataTimeStampTableSize >> 2;    // :4

                            if( (dataTimeStampTableSize64 > 0) && (dataTimeStampTableSize64 < 16) && (dataTimeStampTableSize64 < *timeStampTableSize64) )
                            {
                                memset( (char*)timestampTable, 0, *timeStampTableSize64 );

                                for(unsigned int n = 0; n < dataTimeStampTableSize64; n++)
                                {
                                    uint64_t tsm_1 = dataTable[ data_cnt++ ] & 0x000000000000FFFF;
                                    uint64_t tsm_2 = dataTable[ data_cnt++ ] & 0x000000000000FFFF;
                                    uint64_t tsm_3 = dataTable[ data_cnt++ ] & 0x000000000000FFFF;
                                    uint64_t tsm_4 = dataTable[ data_cnt++ ] & 0x000000000000FFFF;

                                    timestampTable[ n ] = (tsm_1 << 0) | (tsm_2 << 16) | (tsm_3 << 32) | (tsm_4 << 48);
                                }

                                // Zapisujemy rzeczywisty rozmaiar stepla czasowego
                                *timeStampTableSize64 = dataTimeStampTableSize64;

                                int dataTableSize = dataSize - 2 - dataTimeStampTableSize;

                                // Kopiowanie danych do tablicy wyjsciowej
                                if( dataTableSize > 0 )
                                {
                                    for( unsigned int n = 0; n < (unsigned int)dataTableSize; n++ )
                                    {
                                        data[ n ] = dataTable[ data_cnt++ ];
                                    }
                                }
                            }
                            else
                            {
                                LOGGER( LOG_ERROR << "Timestamp Error: " << dataTimeStampTableSize );
                                isError = true;
                            }
                        }
                        else
                        {
                            isError = true;
                        }
                    }
                    else
                    {
                        isError = true;
                    }
                }
                else
                {
                    isError = true;
                }
            }
            else
            {
                isError = true;
            }
        }
        else
        {
            isError = true;
        }

        if( isError )
        {
            m_STAT_DataSizeError++;
            return false;
        }
        else
        {
            m_STAT_CommunicationSucces++;
        }

        return true;
    }

}
