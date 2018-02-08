#ifndef COMMUNICATIONINTERFACE_H
#define COMMUNICATIONINTERFACE_H

#include <stdint.h>

namespace codeframe
{

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class CommunicationInterface
{
public:
    virtual uint64_t Write( int id, const char *data, uint64_t len ) = 0;
    virtual uint64_t Read( int id, char *data, uint64_t maxlen ) = 0;
};

}

#endif // COMMUNICATIONINTERFACE_H
