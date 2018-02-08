#ifndef HASHUTILITIES_H
#define HASHUTILITIES_H

#include <QString>
#include <QCryptographicHash>

static QString HashString( QString string )
{
    QByteArray datatext = string.toUtf8().left(1000);

    QString encodedPass = QString(QCryptographicHash::hash((datatext),QCryptographicHash::Sha256));
    return encodedPass;
}

#endif // HASHUTILITIES_H
