#ifndef CONSTELEMENTSFACTORY_HPP_INCLUDED
#define CONSTELEMENTSFACTORY_HPP_INCLUDED

#include <sigslot.h>

#include <serializable.hpp>
#include <serializablecontainer.hpp>
#include <extpoint2d.hpp>

#include "constelement.hpp"

class ConstElementsFactory : public codeframe::cSerializableContainer
{
        CODEFRAME_META_CLASS_NAME( "ConstElementsFactory" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
        ConstElementsFactory( const std::string& name, ObjectNode* parent );
        virtual ~ConstElementsFactory();

        smart_ptr<ConstElement> Create( smart_ptr<ConstElement> );
        smart_ptr<ConstElement> CreateLine( codeframe::Point2D<int> sPoint, codeframe::Point2D<int> ePoint );

        signal1< smart_ptr<ConstElement> > signalElementAdd;
        signal1< smart_ptr<ConstElement> > signalElementDel;

    protected:
        smart_ptr<codeframe::ObjectNode> Create(
                                                 const std::string& className,
                                                 const std::string& objName,
                                                 const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                               );

    private:

};

#endif // CONSTELEMENTSFACTORY_HPP_INCLUDED
