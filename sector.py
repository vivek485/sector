import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, timedelta
import numpy as np
import ta
import time
import asyncio
import aiohttp
import logging
import hashlib
from functools import lru_cache
import streamlit as st
from contextlib import contextmanager
from asyncio import Semaphore
from aiohttp import TCPConnector
import nest_asyncio
from sectordata import auto,niftybank,energy,finance,fmcg,it,media,metal,pharma,psu,health,consumer,oil,next50,niftymidcap,fno

from asyncio import Semaphore
st.set_page_config(layout="wide")
st.title("sector Scanner")
sec = ['auto','niftybank','energy','finance','fmcg','it','media','metal','pharma','psu','health','consumer','oil','next50','niftymidcap','fno']
selectsec = st.selectbox('sector',options=sec)

if selectsec == 'auto':
    s=auto
if selectsec == 'niftybank':
    s=niftybank
if selectsec == 'energy':
    s=energy
if selectsec == 'finance':
    s=finance
if selectsec == 'fmcg':
    s=fmcg
if selectsec == 'it':
    s=it
if selectsec == 'media':
    s=media
if selectsec == 'metal':
    s=metal
if selectsec == 'pharma':
    s=pharma
if selectsec == 'psu':
    s=psu
if selectsec == 'health':
    s=health
if selectsec == 'consumer':
    s=consumer
if selectsec == 'oil':
    s=oil
if selectsec == 'next50':
    s=next50
if selectsec == 'niftymidcap':
    s=niftymidcap
if selectsec == 'fno':
    s=fno

# Configuration
CONFIG = {
    'interval':5,
    'dayback': 50,
    'timezone': 'Asia/Kolkata',
    'batch_size': 50,
    'semaphore_limit': 10
}

# Initialize global variables
ist_timezone = pytz.timezone(CONFIG['timezone'])
ed = datetime.now()
stdate = ed - timedelta(days=CONFIG['dayback'])

# Cache implementation
class DataCache:
    def __init__(self):
        self.cache = {}
        self.max_age = timedelta(minutes=15)

    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.max_age:
                return data
            else:
                del self.cache[key]
        return None

    def set(self, key, data):
        self.cache[key] = (data, datetime.now())

data_cache = DataCache()

def conv(x):
    timestamp = int(x.timestamp() * 1000)
    timestamp_str = str(timestamp)[:-4] + '0000'
    return int(timestamp_str)

@lru_cache(maxsize=1000)
def get_cache_key(stock, fromdate, todate, interval):
    return hashlib.md5(f"{stock}{fromdate}{todate}{interval}".encode()).hexdigest()

fromdate = conv(stdate)
todate = conv(ed)
sem = Semaphore(CONFIG['semaphore_limit'])

def plot_stock_chart(df, symbol):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'),
                  row=1, col=1)

    # Add vertical white dashed line at 9:15 candle for each day
    nine_fifteen_times = [idx for idx in df.index if hasattr(idx, 'time') and idx.time() == pd.Timestamp('09:15:00').time()]
    for x_val in nine_fifteen_times:
        fig.add_vline(x=x_val, line_width=1, line_dash="dot", line_color="white", row=1, col=1)

    # Add triangle marker for 9:15 candle if open==low or open==high
    marker_x = []
    marker_y = []
    marker_symbol = []
    marker_color = []
    for idx in nine_fifteen_times:
        row = df.loc[idx]
        if row['Open'] == row['Low']:
            marker_x.append(idx)
            marker_y.append(row['Open'])
            marker_symbol.append('triangle-up')
            marker_color.append('green')
        elif row['Open'] == row['High']:
            marker_x.append(idx)
            marker_y.append(row['Open'])
            marker_symbol.append('triangle-down')
            marker_color.append('red')
    if marker_x:
        fig.add_trace(go.Scatter(
            x=marker_x,
            y=marker_y,
            mode='markers',
            marker=dict(symbol=marker_symbol, color=marker_color, size=16, line=dict(width=1, color='black')),
            name='9:15 O=H/L',
            showlegend=True
        ), row=1, col=1)

    # Add indicators
    fig.add_trace(go.Scatter(x=df.index, y=df['ma100'],
                            line=dict(color='purple', width=1),
                            name='100 MA'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['prhigh'],
                            line=dict(color='red', width=1),
                            name='prhigh'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['prlow'],
                            line=dict(color='green', width=1),
                            name='prlow'),
                  row=1, col=1)

    # Net volume
    if 'net volume' in df.columns:
        colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, 
                            y=df['net volume'],
                            name='Net Volume',
                            marker_color=colors,
                            opacity=0.5),
                      row=2, col=1)
        netvol_sma20 = df['net volume'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=netvol_sma20,
                                 mode='lines',
                                 line=dict(color='white', width=0.5, dash='dot'),
                                 name='Net Volume SMA 20'),
                      row=2, col=1)

    fig.update_layout(
        title=f'{symbol} Price Chart',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=600
    )
    fig.update_xaxes(type='category', row=1, col=1)
    fig.update_xaxes(type='category', row=2, col=1)
    return fig

async def get_previous_day_data(session, stock):
    """Get previous day's high and low data only"""
    async with sem:
        try:
            # Get data for only 2 days to calculate previous day's high-low
            prev_stdate = ed - timedelta(days=2)
            prev_fromdate = conv(prev_stdate)
            prev_todate = conv(ed)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            url = f'https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/{stock}?endTimeInMillis={prev_todate}&intervalInMinutes={CONFIG["interval"]}&startTimeInMillis={prev_fromdate}'
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return None, None, None

                resp = await response.json()
                if not resp.get('candles'):
                    return None, None, None

                candle = resp['candles']
                dt = pd.DataFrame(candle)
                if dt.empty:
                    return None, None, None

                fd = dt.rename(columns={0: 'datetime', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'})
                fd['symbol'] = stock
                final_df = fd

                # Process datetime
                final_df['datetime1'] = pd.to_datetime(final_df['datetime'], unit='s', utc=True).dt.tz_convert(ist_timezone)
                final_df.set_index('datetime1', inplace=True)
                final_df.drop(columns=['datetime'], inplace=True)
                
                # Get previous day's date
                today = pd.Timestamp.now(ist_timezone).date()
                prev_date = today - timedelta(days=1)
                
                # Get previous day's data
                prev_day_data = final_df[final_df.index.date == prev_date]
                
                if prev_day_data.empty:
                    return None, None, None
                
                prev_high = prev_day_data['High'].max()
                prev_low = prev_day_data['Low'].min()
                
                # Calculate high-low percentage
                if prev_low > 0:  # Avoid division by zero
                    high_low_percentage = ((prev_high - prev_low) / prev_low) * 100
                else:
                    high_low_percentage = 0
                
                return stock, high_low_percentage, prev_high, prev_low

        except Exception as e:
            st.error(f"Error getting previous day data for {stock}: {str(e)}")
            return None, None, None, None

async def getdata(session, stock):
    async with sem:
        try:
            cache_key = get_cache_key(stock, fromdate, todate, CONFIG['interval'])
            cached_data = data_cache.get(cache_key)
            if cached_data is not None:
                return cached_data

            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            url = f'https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/{stock}?endTimeInMillis={todate}&intervalInMinutes={CONFIG["interval"]}&startTimeInMillis={fromdate}'
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return None

                resp = await response.json()
                if not resp.get('candles'):
                    return None

                candle = resp['candles']
                dt = pd.DataFrame(candle)
                if dt.empty:
                    return None

                fd = dt.rename(columns={0: 'datetime', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'})
                fd['symbol'] = stock
                final_df = fd

                # Process datetime
                final_df['datetime1'] = pd.to_datetime(final_df['datetime'], unit='s', utc=True).dt.tz_convert(ist_timezone)
                final_df.set_index('datetime1', inplace=True)
                final_df.drop(columns=['datetime'], inplace=True)

                # Calculate net volume for each day
                final_df['date'] = final_df.index.date
                final_df['time'] = final_df.index.time
                final_df['net volume'] = np.nan
                for date in final_df['date'].unique():
                    day_mask = final_df['date'] == date
                    day_data = final_df.loc[day_mask, 'Volume']
                    net_vol = day_data.diff()
                    net_vol.iloc[0] = day_data.iloc[0]
                    final_df.loc[day_mask, 'net volume'] = net_vol
                # Drop temporary columns
                final_df.drop(['date', 'time'], axis=1, inplace=True)

                # Technical indicators
                final_df['ema20'] = ta.trend.ema_indicator(final_df['Close'], window=25)
                final_df['atr'] = ta.volatility.average_true_range(final_df['High'], final_df['Low'], final_df['Close'], window=25)
                final_df['kc_lower'] = final_df['ema20'] - (final_df['atr'] * 1)
                final_df['kc_upper'] = final_df['ema20'] + (final_df['atr'] * 1)
                
                # Add 200 MA
                final_df['ma100'] = ta.trend.sma_indicator(final_df['Close'], window=100)
                final_df['ma50'] = ta.trend.sma_indicator(final_df['Close'], window=50)
                final_df['ma20'] = ta.trend.sma_indicator(final_df['Close'], window=20)
                final_df['hlc/3'] = (final_df[['High', 'Low', 'Close']].sum(axis=1)) / 3
                final_df['ema_hlc/3'] = ta.trend.sma_indicator(final_df['hlc/3'], window=5)

                # Calculate Bollinger Bands
                final_df['bb_middle'] = ta.trend.sma_indicator(final_df['Close'], window=20)
                final_df['bb_std'] = final_df['Close'].rolling(window=20).std()
                final_df['bb_upper'] = final_df['bb_middle'] + (final_df['bb_std'] * 2)
                final_df['bb_lower'] = final_df['bb_middle'] - (final_df['bb_std'] * 2)
                final_df['time'] = final_df.index.time

                # Filter rows where the time component is 09:15:00
                nine_fifteen_df = final_df[final_df['time'] == pd.Timestamp('09:15:00').time()].iloc[-2]

                # Get the date from nine_fifteen_df
                prev_date = nine_fifteen_df.name.date()

                # Create a copy of data for the previous date
                prev_date_df = final_df[final_df.index.date == prev_date].copy()

                previouslow = prev_date_df['Low'].min()
                previoushigh = prev_date_df['High'].max()
                final_df['prhigh'] = previoushigh
                final_df['prlow'] = previouslow
                final_df['prevclose'] = final_df['Close'].shift(1)
                
                final_df=final_df.iloc[-120:]
                data_cache.set(cache_key, final_df)
                return final_df

        except Exception as e:
            st.error(f"Error processing {stock}: {str(e)}")
            return None

async def process_batch(session, stocks_batch):
    tasks = [getdata(session, stock) for stock in stocks_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def get_previous_day_batch(session, stocks_batch):
    """Process batch for previous day data"""
    tasks = [get_previous_day_data(session, stock) for stock in stocks_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def get_sorted_stocks():
    """Get stocks sorted by previous day's high-low percentage in ascending order"""
    timeout = aiohttp.ClientTimeout(total=30)
    connector = TCPConnector(limit=50, force_close=True)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        stock_percentages = []
        
        # Get previous day data for all stocks
        for i in range(0, len(s), CONFIG['batch_size']):
            batch = s[i:i + CONFIG['batch_size']]
            batch_results = await get_previous_day_batch(session, batch)
            
            for result in batch_results:
                if (result is not None and not isinstance(result, Exception) 
                    and result[0] is not None and result[1] is not None):
                    stock, percentage, prev_high, prev_low = result
                    stock_percentages.append((stock, percentage, prev_high, prev_low))
            
            await asyncio.sleep(1)
        
        # Sort by high-low percentage in ascending order (lowest percentage first)
        stock_percentages.sort(key=lambda x: x[1])
        
        # Create sorted stock list
        sorted_stocks = [item[0] for item in stock_percentages]
        
        # Display the sorted list with percentages
        st.subheader("Stocks Sorted by Previous Day High-Low Percentage (Ascending)")
        display_data = []
        for stock, percentage, prev_high, prev_low in stock_percentages:
            display_data.append({
                'Stock': stock,
                'Prev Day High': round(prev_high, 2),
                'Prev Day Low': round(prev_low, 2),
                'High-Low %': round(percentage, 2)
            })
        
        if display_data:
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)
        
        return sorted_stocks

async def main_sorted(sorted_stocks):
    """Run main analysis with sorted stocks"""
    timeout = aiohttp.ClientTimeout(total=30)
    connector = TCPConnector(limit=50, force_close=True)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        results = []
        for i in range(0, len(sorted_stocks), CONFIG['batch_size']):
            batch = sorted_stocks[i:i + CONFIG['batch_size']]
            batch_results = await process_batch(session, batch)
            results.extend([r for r in batch_results if r is not None and not isinstance(r, Exception)])
            await asyncio.sleep(1)
        return results

# Create a wrapper function to run async code
def run_async_code():
    """Wrapper function to run async code in synchronous context"""
    nest_asyncio.apply()
    
    # Step 1: Get sorted stocks based on previous day high-low percentage
    sorted_stocks = asyncio.run(get_sorted_stocks())
    
    if not sorted_stocks:
        st.error("No stock data available for sorting")
        return
    
    st.success(f"Sorted {len(sorted_stocks)} stocks by previous day high-low percentage")
    
    # Step 2: Run main analysis with sorted stocks
    st.info("Now running main analysis with sorted stocks...")
    results = asyncio.run(main_sorted(sorted_stocks))
    
    st.success("Scan complete!")
    
    # Display results in the sorted order
    col1, col2 = st.columns(2)
    
    buy_stocks = []
    sell_stocks = []

    for stock_data in results:
        if stock_data is not None:
            # Use the last row for the latest values
            last_row = stock_data.iloc[-1]
            prevclose = last_row['prevclose']
            prhigh = last_row['prhigh']
            prlow = last_row['prlow']
            symbol = last_row['symbol']

            if prevclose > prhigh:
                buy_stocks.append((stock_data, symbol))
            elif prevclose < prlow:
                sell_stocks.append((stock_data, symbol))

    # Plot buy stocks in sorted order (lowest percentage first)
    with col1:
        st.subheader("Buy Signals (Ascending Order - Lowest % First)")
        for stock_data, symbol in buy_stocks:
            fig = plot_stock_chart(stock_data, symbol)
            st.plotly_chart(fig, use_container_width=True)

    # Plot sell stocks in sorted order (lowest percentage first)
    with col2:
        st.subheader("Sell Signals (Ascending Order - Lowest % First)")
        for stock_data, symbol in sell_stocks:
            fig = plot_stock_chart(stock_data, symbol)
            st.plotly_chart(fig, use_container_width=True)

    # Show info if no signals found
    if not buy_stocks and not sell_stocks:
        st.info("No signals found for current session")

if st.button("Get Sorted Signals"):
    with st.spinner("First, getting previous day data and sorting stocks..."):
        run_async_code()

st.sidebar.markdown("---")
st.sidebar.write("Settings:")
st.sidebar.write(f"Interval: {CONFIG['interval']} min")
st.sidebar.write(f"Lookback: {CONFIG['dayback']} days")
