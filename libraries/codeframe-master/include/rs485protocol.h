#ifndef CRS485PROTOCOL_H
#define CRS485PROTOCOL_H

#include <communicationinterface.h>
#include <serializableproperty.h>

#define USART_RX_BUFFER_DATA_SIZE 16383
#define TIMESTAMP_BUFFER_DATA_SIZE 128

#define REG_MODE  0x00
#define REG_START 0x01
#define REG_CNT   0x02
#define REG_DATA0 0x03

#define HEADER_8ByteSize 6
#define HEADER_16ByteSize 3

namespace codeframe
{
    class cSerialProtocol
    {
    public:
         cSerialProtocol( CommunicationInterface* comdrv, int id, uint8_t addr );   // QT constructor
        ~cSerialProtocol();

         bool IsValid() const;

         bool Synchronize     ( Property* prop, bool forceRead = false, bool forceWrite = false );
         bool SynchronizeRange( Property* prop, uint16_t* data, uint16_t dataSize, uint64_t* timestampTable, uint16_t* timeStampTableSize64 );

    private:
        uint8_t m_thisAdr;

        CommunicationInterface* m_comdrv;

        const uint8_t c_HEADER_SIZE;

        int m_id;

        uint16_t dataTable[ USART_RX_BUFFER_DATA_SIZE ];
        uint16_t dataSize;

        uint32_t m_STAT_DataSizeError;
        uint32_t m_STAT_CommunicationError;
        uint32_t m_STAT_CommunicationSucces;
    };
}

#endif // CRS485PROTOCOL_H
