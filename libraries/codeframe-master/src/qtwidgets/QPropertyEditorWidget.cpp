#include "QPropertyEditorWidget.h"
#include "QPropertyModel.h"
#include "QVariantDelegate.h"
#include "QProperty.h"

QPropertyEditorWidget::QPropertyEditorWidget(QWidget* parent /*= 0*/) : QTreeView(parent)
{
	m_model = new QPropertyModel(this);		
	setModel(m_model);
	setItemDelegate(new QVariantDelegate(this));
}


QPropertyEditorWidget::~QPropertyEditorWidget()
{
}

void QPropertyEditorWidget::addObject( codeframe::cSerializable* propertyObject )
{
	m_model->addItem(propertyObject);
	expandToDepth(0);

    this->setColumnWidth(0, 140);
}

void QPropertyEditorWidget::setObject( codeframe::cSerializable *propertyObject )
{
	m_model->clear();
	if (propertyObject)
		addObject(propertyObject);
}

void QPropertyEditorWidget::updateObject( codeframe::cSerializable* propertyObject )
{
	if (propertyObject)
		m_model->updateItem(propertyObject);	
}

void QPropertyEditorWidget::registerCustomPropertyCB(UserTypeCB callback)
{
	m_model->registerCustomPropertyCB(callback);
}

void QPropertyEditorWidget::unregisterCustomPropertyCB(UserTypeCB callback)
{
	m_model->unregisterCustomPropertyCB(callback);
}

void QPropertyEditorWidget::Clear()
{
    m_model->clear();
}


