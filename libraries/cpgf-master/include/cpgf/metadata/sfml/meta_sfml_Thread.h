// Auto generated file, don't modify.

#ifndef __META_SFML_THREAD_H
#define __META_SFML_THREAD_H


#include "cpgf/gmetadefine.h"
#include "cpgf/metadata/gmetadataconfig.h"
#include "cpgf/metadata/private/gmetadata_header.h"
#include "cpgf/gmetapolicy.h"


using namespace sf;


namespace meta_sfml { 


template <typename D>
void buildMetaClass_Thread(const cpgf::GMetaDataConfigFlags & config, D _d)
{
    (void)config; (void)_d; (void)_d;
    using namespace cpgf;
    
    _d.CPGF_MD_TEMPLATE _constructor<void * (Thread::FuncType, void *)>()
        ._default(copyVariantFromCopyable((void *)NULL))
    ;
    _d.CPGF_MD_TEMPLATE _method("Launch", &D::ClassType::Launch);
    _d.CPGF_MD_TEMPLATE _method("Wait", &D::ClassType::Wait);
    _d.CPGF_MD_TEMPLATE _method("Terminate", &D::ClassType::Terminate);
}


} // namespace meta_sfml




#include "cpgf/metadata/private/gmetadata_footer.h"


#endif
