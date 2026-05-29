import os
import glob
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s- %(message)s')
def process_raw_file(file_path):
    logging.info(f"Ingesting raw file:{file_path}")
    try:
        df=pd.read_csv(file_path,parse_dates=['Date'],index_col='Date')
        if df.empty:
            logging.warning(f"File{file_path} is empty. Skipping.")
            return
        df=df.ffill().bfill()
        years_of_data=(df.index.max() - df.index.min()).days/365.25
        if years_of_data<5.0:
            logging.error(f"Validation Failed: {file_path} only has {years_of_data:.2f}years of data.")
            return
        base_name=os.path.basename(file_path).replace(".csv",".parquet")
        output_path=os.path.join("data","processed",base_name)
        df= df.astype('float32')
        df.to_parquet(output_path,compression='snappy')
        logging.info(f"Successfully processed and saved: {output_path}(span:{years_of_data:.2f}years)")
    except Exception as e:
        logging.erro(f"Failed to process file{file_path}:{e}")
def run_ingestion_pipeline():
    raw_files=glob.glob(os.path.join("data","raw","*.csv"))
    if not raw_files:
        logging.warning("No raw CSV files found.Please execute the downloader first.")
        return
    for file_path in raw_files:
        process_raw_file(file_path)
if __name__=="__main__":
    run_ingestion_pipeline() 