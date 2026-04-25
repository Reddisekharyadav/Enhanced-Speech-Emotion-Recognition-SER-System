import argparse
import sys
import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1440,1600")

    service = Service()
    return webdriver.Chrome(service=service, options=options)


def ping_app(url: str, wait_seconds: int) -> None:
    driver = build_driver()
    try:
        print(f"Opening {url}")
        driver.set_page_load_timeout(60)
        driver.get(url)

        # Wait until the document is loaded and Streamlit has rendered a page body.
        WebDriverWait(driver, 45).until(
            lambda current_driver: current_driver.execute_script("return document.readyState") == "complete"
        )
        WebDriverWait(driver, 45).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        title = driver.title.strip() or "Untitled page"
        print(f"Page loaded: {title}")
        print(f"Waiting {wait_seconds} seconds to keep the session warm...")
        time.sleep(wait_seconds)
        print("Keep-alive completed successfully.")
    except TimeoutException as exc:
        print(f"Timed out while loading {url}: {exc}", file=sys.stderr)
        raise
    except WebDriverException as exc:
        print(f"Selenium could not open the app: {exc}", file=sys.stderr)
        raise
    finally:
        driver.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep a Streamlit app alive with Selenium.")
    parser.add_argument("--url", required=True, help="Streamlit app URL to open")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=12,
        help="How many seconds to keep the page open after it loads",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ping_app(args.url, args.wait_seconds)
