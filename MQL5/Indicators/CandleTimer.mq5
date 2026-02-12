//+------------------------------------------------------------------+
//|                                                  CandleTimer.mq5 |
//|                                  Copyright 2023, Forex_DNN Agent |
//|                                             https://github.com/  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Forex_DNN Agent"
#property link      "https://github.com/"
#property version   "1.00"
#property indicator_chart_window
#property indicator_plots 0

//--- input parameters
input group      "--- Display Settings ---"
input color      InpFontColor      = clrYellow;          // Font Color
input int        InpFontSize       = 12;                 // Font Size
input string     InpFontName       = "Arial";            // Font Name
input ENUM_BASE_CORNER InpCorner   = CORNER_RIGHT_UPPER; // Chart Corner
input int        InpXOffset        = 10;                 // X Offset
input int        InpYOffset        = 20;                 // Y Offset

input group      "--- Alert Settings ---"
input int        InpAlertSeconds   = 3;                  // Seconds before candle close to alert
input bool       InpEnableAlert    = true;               // Enable Popup Alert
input bool       InpEnablePush     = false;              // Enable Push Notification
input bool       InpEnableSound    = true;               // Enable Sound Alert
input string     InpSoundFile      = "alert.wav";        // Sound File Name

//--- global variables
string obj_name = "CandleTimerLabel";
datetime last_alert_time = 0;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- create label
   if(!ObjectCreate(0, obj_name, OBJ_LABEL, 0, 0, 0))
   {
      Print("Failed to create the object! Error code: ", GetLastError());
      return(INIT_FAILED);
   }

   ObjectSetInteger(0, obj_name, OBJPROP_CORNER, InpCorner);
   ObjectSetInteger(0, obj_name, OBJPROP_XDISTANCE, InpXOffset);
   ObjectSetInteger(0, obj_name, OBJPROP_YDISTANCE, InpYOffset);
   ObjectSetInteger(0, obj_name, OBJPROP_COLOR, InpFontColor);
   ObjectSetInteger(0, obj_name, OBJPROP_FONTSIZE, InpFontSize);
   ObjectSetString(0, obj_name, OBJPROP_FONT, InpFontName);
   ObjectSetString(0, obj_name, OBJPROP_TEXT, "Initializing...");
   ObjectSetInteger(0, obj_name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, obj_name, OBJPROP_HIDDEN, true);

   //--- set timer to update every second
   EventSetTimer(1);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   ObjectDelete(0, obj_name);
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
   //--- update on every tick
   UpdateTimer();
   return(rates_total);
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   UpdateTimer();
}

//+------------------------------------------------------------------+
//| Update the candle timer display and check for alerts             |
//+------------------------------------------------------------------+
void UpdateTimer()
{
   // TimeTradeServer() returns the estimated server time.
   datetime now = TimeTradeServer();

   datetime candle_start = 0;
   if(!SeriesInfoInteger(Symbol(), Period(), SERIES_LASTBAR_DATE, candle_start))
   {
      // Fallback to iTime if SeriesInfoInteger fails
      candle_start = iTime(Symbol(), Period(), 0);
   }

   int period_sec = PeriodSeconds();
   int remaining = (int)(candle_start + period_sec - now);

   if(remaining < 0) remaining = 0;

   //--- Format time
   string time_str = "";
   int hours = remaining / 3600;
   int minutes = (remaining % 3600) / 60;
   int seconds = remaining % 60;

   if(hours > 0)
      time_str = StringFormat("%02d:%02d:%02d", hours, minutes, seconds);
   else
      time_str = StringFormat("%02d:%02d", minutes, seconds);

   ObjectSetString(0, obj_name, OBJPROP_TEXT, time_str);

   //--- Alert check
   if(remaining <= InpAlertSeconds && remaining > 0 && last_alert_time != candle_start)
   {
      string period_str = EnumToString(Period());
      StringReplace(period_str, "PERIOD_", "");
      string msg = StringFormat("Candle closing in %d seconds on %s %s", remaining, Symbol(), period_str);

      if(InpEnableAlert) Alert(msg);
      if(InpEnablePush)  SendNotification(msg);
      if(InpEnableSound) PlaySound(InpSoundFile);

      last_alert_time = candle_start;
   }

   ChartRedraw();
}
//+------------------------------------------------------------------+
