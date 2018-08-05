#ifndef CONSTELEMENTSFACTORY_HPP_INCLUDED
#define CONSTELEMENTSFACTORY_HPP_INCLUDED

#include <sigslot.h>
#include <serializable.h>
#include <serializablecontainer.h>
#include <serializableinterface.h>

#include "constelementline.hpp"

class ConstElementsFactory : public codeframe::cSerializableContainer
{
    public:
        std::string Role()            const { return "Container";            }
        std::string Class()           const { return "ConstElementsFactory"; }
        std::string BuildType()       const { return "Static";               }
        std::string ConstructPatern() const { return ""; }

    public:
        ConstElementsFactory( std::string name, cSerializableInterface* parent );
        virtual ~ConstElementsFactory();

        smart_ptr<ConstElement> Create( int x, int y, int z );

        signal1< smart_ptr<ConstElement> > signalElementAdd;
        signal1< smart_ptr<ConstElement> > signalElementDel;

    protected:
        smart_ptr<codeframe::cSerializableInterface> Create(
                                                             const std::string className,
                                                             const std::string objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                            );

    private:

};

#endif // CONSTELEMENTSFACTORY_HPP_INCLUDED
