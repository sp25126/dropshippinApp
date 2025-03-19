from main import scrape_products
import pandas as pd

if __name__ == "__main__":
    df = scrape_products()
    df.to_csv('products.csv', index=False)
    print("Products updated!")