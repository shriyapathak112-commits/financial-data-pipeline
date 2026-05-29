import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime,timedelta
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s- %(message)s')
def get_sp500_tickers_by_sector():
    try:
        url="https://wikipedia.org"
        table=pd.read_html(url)[0]
        sector_dict=table.groupby('GICS Sector')['Symbol'].apply(list).to_dict()
        return sector_dict
    except Exception as e:
        logging.error(f"Failed to fetch S&P 500 tickers:{e}")
        return {}
def download_asset_group(tickers,filename,start_date,end_date):
    if not tickers:
        return
    logging.info(f"Downloading {len(tickers)} tickers for {filename}...")
    try:
        data=yf.download(tickers,start=start_date, end=end_date, group_by='ticker',threads=True)
        adj_close_df=pd.DataFrame()
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                adj_close_df[ticker]= data[ticker]['Adj Close']
        output_path=os.path.join("data","raw",f"{filename}.csv")
        adj_close_df.to_csv(output_path)
        logging.info(f"Successfully saved raw data to {output_path}")
    except Exception as e:
        logging.error(f"Error downloading asset group{filename}:{e}")
def run_download_pipeline():
    end_date=datetime.today().strftime('%Y-%m-%d')
    start_date=(datetime.today()-timedelta(days=5*365+15)).strftime('%Y-%m-%d')
    sectors=get_sp500_tickers_by_sector()
    for sector,tickers in sectors.items():
        clean_sector_name=sector.lower().replace("","_").replace("&","and")
        download_asset_group(tickers, f"sp500_sector_{clean_sector_name}",start_date,end_date)
    macro_assets= {
        "etfs":["SPY","QQQ","IVV","IWM","EEM"],
        "bonds":["TLT","IEF","SHY","BND"],
        "risk_free_rate":["^IRX"]
    }
    for asset_name,tickers in macro_assets.items():
        download_asset_group(tickers,asset_name,start_date,end_date)
if __name__=="__main__":
    run_download_pipeline()

