#property copyright "Forex_DNN"
#property link      "https://forexdnn.com"
#property version   "2.01"
#property indicator_chart_window
#property indicator_plots 0

// Inputs
input string   InpFilesPrefix        = "";          // Prefix (optional)
input int      InpPollIntervalMs     = 1000;        // Polling interval in ms

// Stored last modified times for files to implement incremental/asynchronous checks
datetime file_last_mod_struct = 0;
datetime file_last_mod_levels = 0;
datetime file_last_mod_zones  = 0;
datetime file_last_mod_sigs   = 0;
datetime file_last_mod_state  = 0;
datetime file_last_mod_ml     = 0;

// Track active objects per layer/file to implement precise incremental stale-object deletion
string tracked_struct[];
string tracked_levels[];
string tracked_zones[];
string tracked_sigs[];
string tracked_state[];
string tracked_ml[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   EventSetMillisecondTimer(InpPollIntervalMs);
   ArrayResize(tracked_struct, 0);
   ArrayResize(tracked_levels, 0);
   ArrayResize(tracked_zones, 0);
   ArrayResize(tracked_sigs, 0);
   ArrayResize(tracked_state, 0);
   ArrayResize(tracked_ml, 0);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   ClearAllVisualObjects();
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
    return(rates_total);
}

//+------------------------------------------------------------------+
//| Timer event handler for polling Python drawings                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   string symbol = _Symbol;

   // Poll each layer independently, perform delta update, and prune stale objects
   PollAndRenderFile(symbol + "_structure.csv", file_last_mod_struct, tracked_struct);
   PollAndRenderFile(symbol + "_levels.csv", file_last_mod_levels, tracked_levels);
   PollAndRenderFile(symbol + "_zones.csv", file_last_mod_zones, tracked_zones);
   PollAndRenderFile(symbol + "_signals.csv", file_last_mod_sigs, tracked_sigs);
   PollAndRenderFile(symbol + "_state.csv", file_last_mod_state, tracked_state);
   PollAndRenderFile(symbol + "_ml.csv", file_last_mod_ml, tracked_ml);

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| Clear all visual objects on deinit                               |
//+------------------------------------------------------------------+
void ClearAllVisualObjects()
{
   ObjectsDeleteAll(0, "FXDNN_");
}

//+------------------------------------------------------------------+
//| Helper to check if a string exists in an array                   |
//+------------------------------------------------------------------+
bool ArrayContains(const string &arr[], string value)
{
   int size = ArraySize(arr);
   for(int i = 0; i < size; i++)
   {
      if(arr[i] == value) return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Synchronize current drawn objects with previous, deleting stale  |
//+------------------------------------------------------------------+
void SyncLayerObjects(const string &current_drawn[], string &previously_tracked[])
{
   int prev_size = ArraySize(previously_tracked);
   for(int i = 0; i < prev_size; i++)
   {
      string obj_name = previously_tracked[i];
      if(!ArrayContains(current_drawn, obj_name))
      {
         ObjectDelete(0, obj_name);
      }
   }

   // Update the tracked objects array to the current set
   int cur_size = ArraySize(current_drawn);
   ArrayResize(previously_tracked, cur_size);
   for(int i = 0; i < cur_size; i++)
   {
      previously_tracked[i] = current_drawn[i];
   }
}

//+------------------------------------------------------------------+
//| Poll file and render if modified                                 |
//+------------------------------------------------------------------+
void PollAndRenderFile(string filename, datetime &last_mod, string &tracked_array[])
{
   if(!FileIsExist(filename))
      return;

   datetime mod_time = (datetime)FileGetInteger(filename, FILE_MODIFY_DATE);
   if(mod_time == last_mod)
      return; // No changes, skip redraw for this layer!

   last_mod = mod_time;

   int file_handle = FileOpen(filename, FILE_READ|FILE_SHARE_READ|FILE_TXT|FILE_ANSI);
   if(file_handle == INVALID_HANDLE)
      return;

   // Read CSV rows
   string header = FileReadString(file_handle); // Skip header row

   string current_drawn[];
   ArrayResize(current_drawn, 0);

   while(!FileIsEnding(file_handle))
   {
      string line = FileReadString(file_handle);
      if(line == "") continue;

      ProcessCSVLine(line, current_drawn);
   }
   FileClose(file_handle);

   // Eliminate stale objects that exist in previous render but not in the current one
   SyncLayerObjects(current_drawn, tracked_array);
}

//+------------------------------------------------------------------+
//| Helper to parse and draw single CSV row                          |
//+------------------------------------------------------------------+
void ProcessCSVLine(string line, string &current_drawn[])
{
   string parts[];
   int count = StringSplit(line, ',', parts);
   if(count < 2) return;

   string type_name = parts[0];
   string name      = parts[1];

   string time1_str = (count > 2) ? parts[2] : "";
   string time2_str = (count > 3) ? parts[3] : "";
   string price1_str = (count > 4) ? parts[4] : "";
   string price2_str = (count > 5) ? parts[5] : "";
   string color_str  = (count > 6) ? parts[6] : "";
   string style_str  = (count > 7) ? parts[7] : "";
   string text       = (count > 8) ? parts[8] : "";

   datetime t1 = (time1_str != "") ? StringToTime(time1_str) : 0;
   datetime t2 = (time2_str != "") ? StringToTime(time2_str) : 0;
   double p1 = (price1_str != "") ? StringToDouble(price1_str) : 0.0;
   double p2 = (price2_str != "") ? StringToDouble(price2_str) : 0.0;
   color col = ParseColor(color_str);

   // Determine object drawing based on type_name
   if(type_name == "LEVEL")
   {
      if(ObjectFind(0, name) < 0)
      {
         ObjectCreate(0, name, OBJ_HLINE, 0, 0, p1);
      }
      else
      {
         ObjectMove(0, name, 0, 0, p1);
      }
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetInteger(0, name, OBJPROP_STYLE, ParseStyle(style_str));
      ObjectSetString(0, name, OBJPROP_TEXT, text);

      int size = ArraySize(current_drawn);
      ArrayResize(current_drawn, size + 1);
      current_drawn[size] = name;
   }
   else if(type_name == "ZONE")
   {
      // ZONE rectangle: lower is p1, upper is p2, t1 is start time
      // To draw rectangle across the chart to current time, use time2 as TimeCurrent or t2
      datetime right_t = (t2 != 0) ? t2 : TimeCurrent();
      if(ObjectFind(0, name) < 0)
      {
         ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, right_t, p2);
      }
      else
      {
         ObjectMove(0, name, 0, t1, p1);
         ObjectMove(0, name, 1, right_t, p2);
      }
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetInteger(0, name, OBJPROP_FILL, true);
      ObjectSetInteger(0, name, OBJPROP_BACK, true); // draw in background
      ObjectSetString(0, name, OBJPROP_TEXT, text);

      int size = ArraySize(current_drawn);
      ArrayResize(current_drawn, size + 1);
      current_drawn[size] = name;
   }
   else if(type_name == "SWING" || type_name == "SIGNAL")
   {
      // Draw arrows
      int arrow_code = (style_str == "ArrowUp") ? 233 : 234;
      datetime arrow_t = (t1 != 0) ? t1 : TimeCurrent();
      double arrow_p = (p1 != 0.0) ? p1 : SymbolInfoDouble(_Symbol, SYMBOL_BID);

      if(ObjectFind(0, name) < 0)
      {
         ObjectCreate(0, name, OBJ_ARROW, 0, arrow_t, arrow_p);
      }
      else
      {
         ObjectMove(0, name, 0, arrow_t, arrow_p);
      }
      ObjectSetInteger(0, name, OBJPROP_ARROWCODE, arrow_code);
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetString(0, name, OBJPROP_TEXT, text);

      int size = ArraySize(current_drawn);
      ArrayResize(current_drawn, size + 1);
      current_drawn[size] = name;
   }
   else if(type_name == "BOS" || type_name == "CHOCH")
   {
      // Draw horizontal segment or ray
      if(ObjectFind(0, name) < 0)
      {
         ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, TimeCurrent(), p1);
      }
      else
      {
         ObjectMove(0, name, 0, t1, p1);
         ObjectMove(0, name, 1, TimeCurrent(), p1);
      }
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetInteger(0, name, OBJPROP_STYLE, ParseStyle(style_str));
      ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, true);
      ObjectSetString(0, name, OBJPROP_TEXT, text);

      int size = ArraySize(current_drawn);
      ArrayResize(current_drawn, size + 1);
      current_drawn[size] = name;
   }
   else if(type_name == "PANEL")
   {
      // Draw standard state panel on top left/right corner
      DrawPanelText(name, text, current_drawn);
   }
}

//+------------------------------------------------------------------+
//| Draw info panel on chart                                         |
//+------------------------------------------------------------------+
void DrawPanelText(string name, string text, string &current_drawn[])
{
   string lines[];
   int line_count = StringSplit(text, ';', lines);
   int y_start = 20;
   if(name == "FXDNN_PANEL_ML") y_start = 120; // separate panel offsets

   for(int i = 0; i < line_count; i++)
   {
      string obj_name = name + "_L" + IntegerToString(i);
      if(ObjectFind(0, obj_name) < 0)
      {
         ObjectCreate(0, obj_name, OBJ_LABEL, 0, 0, 0);
      }
      ObjectSetInteger(0, obj_name, OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, obj_name, OBJPROP_YDISTANCE, y_start + (i * 18));
      ObjectSetInteger(0, obj_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetString(0, obj_name, OBJPROP_TEXT, lines[i]);
      ObjectSetInteger(0, obj_name, OBJPROP_COLOR, clrWhite);

      int size = ArraySize(current_drawn);
      ArrayResize(current_drawn, size + 1);
      current_drawn[size] = obj_name;
   }
}

//+------------------------------------------------------------------+
//| Parse color from string                                          |
//+------------------------------------------------------------------+
color ParseColor(string col_str)
{
   if(col_str == "Red" || col_str == "LightCoral") return clrRed;
   if(col_str == "Blue" || col_str == "LightBlue") return clrBlue;
   if(col_str == "Green") return clrGreen;
   if(col_str == "DarkGreen") return clrDarkGreen;
   if(col_str == "DarkRed") return clrDarkRed;
   if(col_str == "Cyan") return clrCyan;
   if(col_str == "Orange") return clrOrange;
   if(col_str == "Magenta") return clrMagenta;
   return clrGray;
}

//+------------------------------------------------------------------+
//| Parse line style from string                                     |
//+------------------------------------------------------------------+
int ParseStyle(string style_str)
{
   if(style_str == "Dash") return STYLE_DASH;
   if(style_str == "Dot") return STYLE_DOT;
   if(style_str == "DashDot") return STYLE_DASHDOT;
   return STYLE_SOLID;
}
