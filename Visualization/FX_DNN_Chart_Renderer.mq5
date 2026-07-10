//+------------------------------------------------------------------+
//|                                       FX_DNN_Chart_Renderer.mq5  |
//|                                  Copyright 2026, Forex_DNN       |
//|                                             https://forexdnn.com |
//+------------------------------------------------------------------+
#property copyright "Forex_DNN"
#property link      "https://forexdnn.com"
#property version   "1.00"
#property indicator_chart_window
#property indicator_plots 0

input string   InpJsonFileNamePrefix = "FX_DNN_draw_data_";
input int      InpPollIntervalMs     = 1000;

datetime last_timestamp = 0;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   EventSetMillisecondTimer(InpPollIntervalMs);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   ObjectsDeleteAll(0, "FX_DNN_");
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
   string filename = InpJsonFileNamePrefix + _Symbol + ".json";

   if(!FileIsExist(filename))
      return;

   int file_handle = FileOpen(filename, FILE_READ|FILE_SHARE_READ|FILE_TXT|FILE_ANSI);
   if(file_handle == INVALID_HANDLE)
      return;

   string json_content = "";
   while(!FileIsEnding(file_handle))
   {
      json_content += FileReadString(file_handle);
   }
   FileClose(file_handle);

   // Clear previous drawings of prefix "FX_DNN_"
   ObjectsDeleteAll(0, "FX_DNN_");

   // Basic parser for structural coordinates
   // Swings, Zones, and Trade levels are rendered here
   // Using standard MT5 ObjectCreate (OBJ_RECTANGLE, OBJ_ARROW, OBJ_HLINE)

   DrawStatusText(json_content);
   DrawLevels(json_content);

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| Helper to render levels on chart                                 |
//+------------------------------------------------------------------+
void DrawLevels(string json)
{
   // Parse sl_price, tp_price, entry_price from JSON
   double entry = GetDoubleKey(json, "entry_price");
   double sl = GetDoubleKey(json, "sl_price");
   double tp = GetDoubleKey(json, "tp_price");

   if(entry > 0) DrawHorizontalLine("FX_DNN_Entry", entry, clrBlue, STYLE_SOLID);
   if(sl > 0)    DrawHorizontalLine("FX_DNN_SL", sl, clrRed, STYLE_DASH);
   if(tp > 0)    DrawHorizontalLine("FX_DNN_TP", tp, clrGreen, STYLE_DASH);
}

void DrawStatusText(string json)
{
   string regime = GetStringKey(json, "regime");
   string trend = GetStringKey(json, "trend");

   if(regime != "")
   {
      string text = "Regime: " + regime + " | Trend: " + trend;
      ObjectCreate(0, "FX_DNN_Status", OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, "FX_DNN_Status", OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, "FX_DNN_Status", OBJPROP_YDISTANCE, 20);
      ObjectSetInteger(0, "FX_DNN_Status", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetString(0, "FX_DNN_Status", OBJPROP_TEXT, text);
      ObjectSetInteger(0, "FX_DNN_Status", OBJPROP_COLOR, clrWheat);
   }
}

void DrawHorizontalLine(string name, double price, color col, int style)
{
   ObjectCreate(0, name, OBJ_HLINE, 0, 0, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, col);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
}

// Utility JSON helpers
double GetDoubleKey(string json, string key)
{
   int pos = StringFind(json, "\"" + key + "\"", 0);
   if(pos == -1) return 0.0;
   int val_pos = StringFind(json, ":", pos);
   if(val_pos == -1) return 0.0;
   int end_pos = StringFind(json, ",", val_pos);
   if(end_pos == -1) end_pos = StringFind(json, "}", val_pos);
   string sub = StringSubstr(json, val_pos + 1, end_pos - val_pos - 1);
   StringTrimLeft(sub);
   StringTrimRight(sub);
   return StringToDouble(sub);
}

string GetStringKey(string json, string key)
{
   int pos = StringFind(json, "\"" + key + "\"", 0);
   if(pos == -1) return "";
   int val_pos = StringFind(json, ":", pos);
   if(val_pos == -1) return "";
   int start_quote = StringFind(json, "\"", val_pos);
   if(start_quote == -1) return "";
   int end_quote = StringFind(json, "\"", start_quote + 1);
   if(end_quote == -1) return "";
   return StringSubstr(json, start_quote + 1, end_quote - start_quote - 1);
}
