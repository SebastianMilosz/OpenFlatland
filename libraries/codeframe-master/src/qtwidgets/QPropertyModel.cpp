// *************************************************************************************************
//
// QPropertyEditor v 0.3
//   
// --------------------------------------
// Copyright (C) 2007 Volker Wiendl
// Acknowledgements to Roman alias banal from qt-apps.org for the Enum enhancement
//
//
// The QPropertyEditor Library is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation version 3 of the License 
//
// The Horde3D Scene Editor is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// *************************************************************************************************

#include "QPropertyModel.h"

#include "QProperty.h"

#include <QApplication>
#include <QItemEditorFactory>
#include <serializable.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QPropertyModel::OnPropertyChanged( codeframe::Property* prop )
{
    (void)prop;

    layoutChanged();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QPropertyModel::QPropertyModel(QObject* parent /*= 0*/) : QAbstractItemModel(parent)
{	
    m_rootItem = new QProperty( "Root", 0, this );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QPropertyModel::~QPropertyModel()
{
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QModelIndex QPropertyModel::index ( int row, int column, const QModelIndex & parent /*= QModelIndex()*/ ) const
{
    QProperty *parentItem = m_rootItem;
	if (parent.isValid())
        parentItem = static_cast<QProperty*>(parent.internalPointer());
	if (row >= parentItem->children().size() || row < 0)
		return QModelIndex();		
	return createIndex(row, column, parentItem->children().at(row));	
		
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QModelIndex QPropertyModel::parent ( const QModelIndex & index ) const
{
	if (!index.isValid())
		return QModelIndex();

    QProperty *childItem  = static_cast<QProperty*>(index.internalPointer());
    QProperty *parentItem = qobject_cast<QProperty*>(childItem->parent());

	if (!parentItem || parentItem == m_rootItem)
		return QModelIndex();

	return createIndex(parentItem->row(), 0, parentItem);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int QPropertyModel::rowCount ( const QModelIndex & parent /*= QModelIndex()*/ ) const
{
    QProperty *parentItem = m_rootItem;
	if (parent.isValid())
        parentItem = static_cast<QProperty*>(parent.internalPointer());
	return parentItem->children().size();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
int QPropertyModel::columnCount ( const QModelIndex & /*parent = QModelIndex()*/ ) const
{
	return 2;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QVariant QPropertyModel::data ( const QModelIndex & index, int role /*= Qt::DisplayRole*/ ) const
{
	if (!index.isValid())
		return QVariant();

    QProperty *item = static_cast<QProperty*>(index.internalPointer());
	switch(role)
	{
    case Qt::ToolTipRole:
        return item->value(role);
	case Qt::DecorationRole:
	case Qt::DisplayRole:
	case Qt::EditRole:
		if (index.column() == 0)
			return item->objectName().replace('_', ' ');
		if (index.column() == 1)
			return item->value(role);
	case Qt::BackgroundRole:
		if (item->isRoot())	return QApplication::palette("QTreeView").brush(QPalette::Normal, QPalette::Button).color();
		break;
	};
	return QVariant();
}

/*****************************************************************************/
/**
  * @brief edit methods
 **
******************************************************************************/
bool QPropertyModel::setData ( const QModelIndex & index, const QVariant & value, int role /*= Qt::EditRole*/ )
{
	if (index.isValid() && role == Qt::EditRole)
	{
        QProperty *item = static_cast<QProperty*>(index.internalPointer());
		item->setValue(value);
		emit dataChanged(index, index);
		return true;
	}
	return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
Qt::ItemFlags QPropertyModel::flags ( const QModelIndex & index ) const
{
	if (!index.isValid())
		return Qt::ItemIsEnabled;
    QProperty *item = static_cast<QProperty*>(index.internalPointer());
	// only allow change of value attribute
	if (item->isRoot())
		return Qt::ItemIsEnabled;	
	else if (item->isReadOnly())
		return Qt::ItemIsDragEnabled | Qt::ItemIsSelectable;	
	else
		return Qt::ItemIsDragEnabled | Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsEditable;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QVariant QPropertyModel::headerData ( int section, Qt::Orientation orientation, int role /*= Qt::DisplayRole*/ ) const
{
	if (orientation == Qt::Horizontal && role == Qt::DisplayRole) 
	{
		switch (section) 
		{
			 case 0:
				 return tr("Name");
			 case 1:
				 return tr("Value");
		}
	}
	return QVariant();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
QModelIndex QPropertyModel::buddy ( const QModelIndex & index ) const 
{
	if (index.isValid() && index.column() == 0)
		return createIndex(index.row(), 1, index.internalPointer());
	return index;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QPropertyModel::addPropertyRecursivity( QProperty *propertyIteam, codeframe::cSerializable *propertyObject )
{
    // insert properties
    beginInsertRows( QModelIndex(), rowCount(), rowCount() + propertyObject->size() );

    QProperty* propertyRootItem = new QProperty( QString::fromStdString( propertyObject->ObjectName() ), 0, propertyIteam);

    for( codeframe::cSerializable::iterator it = propertyObject->begin(); it != propertyObject->end(); ++it )
    {
        codeframe::Property* iser = *it;

        // Set default name of the hierarchy property to the class name
        QString name = QString::fromStdString( iser->Name() );

        // Create Property Item for class node
        QProperty* propertyItem = new QProperty(name, iser, propertyRootItem);

        // Sodlaczenie sygnalu zmiany danych
        connect( propertyItem, SIGNAL(valueChanged(Property*)), this, SLOT(OnPropertyChanged(Property*)), Qt::QueuedConnection );
    }

    // Po wszystkich obiektach dzieci ladujemy zawartosc
    for( codeframe::cSerializableChildList::iterator it = propertyObject->ChildList()->begin(); it != propertyObject->ChildList()->end(); ++it )
    {
        codeframe::cSerializable* iser = *it;

        addPropertyRecursivity( propertyRootItem, iser );
    }

    endInsertRows();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QPropertyModel::addItem( codeframe::cSerializable *propertyObject )
{
    addPropertyRecursivity( m_rootItem, propertyObject );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QPropertyModel::updateItem( codeframe::cSerializable *propertyObject, const QModelIndex& parent /*= QModelIndex() */ )
{
    (void)propertyObject;
    (void)parent;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QPropertyModel::clear()
{
	beginRemoveRows(QModelIndex(), 0, rowCount());
	delete m_rootItem;
    m_rootItem = new QProperty("Root",0, this);
	endRemoveRows();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QPropertyModel::registerCustomPropertyCB(QPropertyEditorWidget::UserTypeCB callback)
{
	if ( !m_userCallbacks.contains(callback) )
		m_userCallbacks.push_back(callback);
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void QPropertyModel::unregisterCustomPropertyCB(QPropertyEditorWidget::UserTypeCB callback)
{
	int index = m_userCallbacks.indexOf(callback);
	if( index != -1 )
		m_userCallbacks.removeAt(index);
}
