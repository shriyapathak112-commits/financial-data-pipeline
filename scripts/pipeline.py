import logging
from downloader import run_download_pipeline
from ingestion import run_ingestion_pipeline
logging.basicConfig(
    filename='logs/pipeline.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)
def main():
    logging.info("---Starting FINANCIAL DATA PIPELINE---")
    logging.info("Step 1: Launching Automated downloader...")
    run_download_pipeline()
    logging.info("Step 2: Launching Data Ingestion pipeline...")
    run_ingestion_pipeline()
    logging.info("---FINANCIAL DATA PIPELINE COMPLETED---")
if __name__=="__main__":
    main()