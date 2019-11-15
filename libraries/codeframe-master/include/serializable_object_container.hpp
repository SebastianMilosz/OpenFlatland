#ifndef CSERIALIZABLECONTAINER_H
#define CSERIALIZABLECONTAINER_H

#include <exception>
#include <stdexcept>
#include <vector>
#include <cstdbool>

#include <MathUtilities.h>
#include <smartpointer.h>

#include "serializable_object.hpp"
#include "propertyignorelist.hpp"

#define MAXID 100

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    class ObjectContainer : public Object
    {
        friend class cSelectable;

        CODEFRAME_META_CLASS_NAME( "ObjectContainer" );
        CODEFRAME_META_BUILD_ROLE( codeframe::CONTAINER  );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        public:
                     ObjectContainer( const std::string& name, ObjectNode* parentObject );
            virtual ~ObjectContainer();

            virtual smart_ptr<ObjectNode> Create(
                                                  const std::string& className,
                                                  const std::string& objName,
                                                  const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                 ) = 0;

            smart_ptr<ObjectNode> operator[]( int i );

            virtual void CreateRange( const std::string& className, const std::string& objName, int range );
            virtual bool Dispose( unsigned int id );
            virtual bool Dispose( const std::string& objName );
            virtual bool Dispose( smart_ptr<ObjectNode> obj );
            virtual bool DisposeByBuildType( eBuildType serType, cIgnoreList ignore = cIgnoreList() );
            virtual bool Dispose();

            int         Count() const;
            bool        IsName( const std::string& name );
            std::string CreateUniqueName( const std::string& nameBase );
            bool        IsInRange( unsigned int cnt ) const;
            bool        Select( int pos );
            bool        IsSelected();

            smart_ptr<ObjectNode> GetSelected();
            smart_ptr<ObjectNode> Get( int id );

            int Add( smart_ptr<Object> classType, int pos = -1 );

            signal1< smart_ptr<ObjectNode> > signalContainerSelectionChanged;

        protected:
            virtual int InsertObject( smart_ptr<Object> classType, int pos = -1 );

        private:
            void slotSelectionChanged( smart_ptr<ObjectNode> obj );

            std::vector< smart_ptr<Object> > m_containerVector;
            smart_ptr<ObjectNode> m_selected;

            unsigned int m_size;
    };

}

#endif // CSERIALIZABLECONTAINER_H
