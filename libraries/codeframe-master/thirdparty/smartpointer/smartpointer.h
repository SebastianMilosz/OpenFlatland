//#include "yasper.h"
#include <memory>

//#define smart_ptr yasper::ptr
#define smart_ptr std::shared_ptr

//#define smart_isValid(ptr) (ptr.IsValid())
#define smart_ptr_isValid(ptr) (ptr != NULL)

//#define smart_ptr_getRaw(ptr) (ptr)
#define smart_ptr_getRaw(ptr) (ptr.get())

//#define smart_ptr_getCount(ptr) (ptr.GetCount())
#define smart_ptr_getCount(ptr) (ptr.use_count())
