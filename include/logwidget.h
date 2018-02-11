#ifndef LOGWIDGET_H
#define LOGWIDGET_H

#include <imgui.h>
#include <imgui-SFML.h>
#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>

class LogWidget : public sigslot::has_slots<>
{
    public:
        LogWidget();
       ~LogWidget();

        void Clear();
        void AddLog(const char* fmt, ...) IM_FMTARGS(2);
        void Draw(const char* title, bool* p_open = NULL);
        void OnLogMessage(std::string logPath, std::string timestamp, std::string title, std::string msg, int type);

    private:
        ImGuiTextBuffer     Buf;
        ImGuiTextFilter     Filter;
        ImVector<int>       LineOffsets;        // Index to lines offset
        bool                ScrollToBottom;
};

#endif // LOGWIDGET_H
