#ifndef LOGWIDGET_HPP
#define LOGWIDGET_HPP

#include <imgui.h>
#include <imgui-SFML.h>
#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>

class ConsoleWidget : public sigslot::has_slots<>
{
    public:
        ConsoleWidget();
       ~ConsoleWidget();

        void Clear();
        void AddLog(const char* fmt, ...) IM_FMTARGS(2);
        void Draw(const char* title, bool* p_open = NULL);
        void OnLogMessage(const std::string& timestamp, const std::string& title, const std::string& msg, int type);

    private:
        // Portable helpers
        static int   Stricmp(const char* str1, const char* str2)         { int d; while ((d = toupper(*str2) - toupper(*str1)) == 0 && *str1) { str1++; str2++; } return d; }
        static int   Strnicmp(const char* str1, const char* str2, int n) { int d = 0; while (n > 0 && (d = toupper(*str2) - toupper(*str1)) == 0 && *str1) { str1++; str2++; n--; } return d; }
        static char* Strdup(const char *str)                             { size_t len = strlen(str) + 1; void* buf = malloc(len); IM_ASSERT(buf); return (char*)memcpy(buf, (const void*)str, len); }
        static void  Strtrim(char* str)                                  { char* str_end = str + strlen(str); while (str_end > str && str_end[-1] == ' ') str_end--; *str_end = 0; }

        static int TextEditCallbackStub(ImGuiInputTextCallbackData* data) // In C++11 you are better off using lambdas for this sort of forwarding callbacks
        {
            ConsoleWidget* console = (ConsoleWidget*)data->UserData;
            return console->TextEditCallback(data);
        }

        int TextEditCallback(ImGuiInputTextCallbackData* data)
        {
            //AddLog("cursor: %d, selection: %d-%d", data->CursorPos, data->SelectionStart, data->SelectionEnd);
            switch (data->EventFlag)
            {
            case ImGuiInputTextFlags_CallbackCompletion:
                {
                    // Example of TEXT COMPLETION

                    // Locate beginning of current word
                    const char* word_end = data->Buf + data->CursorPos;
                    const char* word_start = word_end;
                    while (word_start > data->Buf)
                    {
                        const char c = word_start[-1];
                        if (c == ' ' || c == '\t' || c == ',' || c == ';')
                            break;
                        word_start--;
                    }

                    // Build a list of candidates
                    ImVector<const char*> candidates;
                    for (int i = 0; i < m_Commands.Size; i++)
                        if (Strnicmp(m_Commands[i], word_start, (int)(word_end-word_start)) == 0)
                            candidates.push_back(m_Commands[i]);

                    if (candidates.Size == 0)
                    {
                        // No match
                        AddLog("No match for \"%.*s\"!\n", (int)(word_end-word_start), word_start);
                    }
                    else if (candidates.Size == 1)
                    {
                        // Single match. Delete the beginning of the word and replace it entirely so we've got nice casing
                        data->DeleteChars((int)(word_start-data->Buf), (int)(word_end-word_start));
                        data->InsertChars(data->CursorPos, candidates[0]);
                        data->InsertChars(data->CursorPos, " ");
                    }
                    else
                    {
                        // Multiple matches. Complete as much as we can, so inputing "C" will complete to "CL" and display "CLEAR" and "CLASSIFY"
                        int match_len = (int)(word_end - word_start);
                        for (;;)
                        {
                            int c = 0;
                            bool all_candidates_matches = true;
                            for (int i = 0; i < candidates.Size && all_candidates_matches; i++)
                                if (i == 0)
                                    c = toupper(candidates[i][match_len]);
                                else if (c == 0 || c != toupper(candidates[i][match_len]))
                                    all_candidates_matches = false;
                            if (!all_candidates_matches)
                                break;
                            match_len++;
                        }

                        if (match_len > 0)
                        {
                            data->DeleteChars((int)(word_start - data->Buf), (int)(word_end-word_start));
                            data->InsertChars(data->CursorPos, candidates[0], candidates[0] + match_len);
                        }

                        // List matches
                        AddLog("Possible matches:\n");
                        for (int i = 0; i < candidates.Size; i++)
                            AddLog("- %s\n", candidates[i]);
                    }

                    break;
                }
            case ImGuiInputTextFlags_CallbackHistory:
                {
                    // Example of HISTORY
                    const int prev_history_pos = m_HistoryPos;
                    if (data->EventKey == ImGuiKey_UpArrow)
                    {
                        if (m_HistoryPos == -1)
                            m_HistoryPos = m_History.Size - 1;
                        else if (m_HistoryPos > 0)
                            m_HistoryPos--;
                    }
                    else if (data->EventKey == ImGuiKey_DownArrow)
                    {
                        if (m_HistoryPos != -1)
                            if (++m_HistoryPos >= m_History.Size)
                                m_HistoryPos = -1;
                    }

                    // A better implementation would preserve the data on the current input line along with cursor position.
                    if (prev_history_pos != m_HistoryPos)
                    {
                        const char* history_str = (m_HistoryPos >= 0) ? m_History[m_HistoryPos] : "";
                        data->DeleteChars(0, data->BufTextLen);
                        data->InsertChars(0, history_str);
                    }
                }
            }
            return 0;
        }

        ImGuiTextBuffer         m_Buf;
        ImGuiTextFilter         m_Filter;
        ImVector<int>           m_LineOffsets;        // Index to lines offset
        bool                    m_ScrollToBottom;
        char                    m_InputBuf[256];
        ImVector<const char*>   m_Commands;
        ImVector<char*>         m_History;
        int                     m_HistoryPos;    // -1: new line, 0..History.Size-1 browsing history.
};

#endif // LOGWIDGET_HPP
