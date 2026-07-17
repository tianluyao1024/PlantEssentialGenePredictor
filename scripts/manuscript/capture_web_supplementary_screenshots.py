"""Capture the public Streamlit interface for the manuscript supplement."""
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


OUT = Path("C:/Users/tly/Downloads/website_screenshots")
OUT.mkdir(parents=True, exist_ok=True)

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1600,1100")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=options)
try:
    driver.get("https://plantessentialgene.com")
    driver.implicitly_wait(10)
    driver.save_screenshot(str(OUT / "website_raw_upload.png"))
    print(driver.title)
finally:
    driver.quit()
