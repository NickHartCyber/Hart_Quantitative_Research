from Hart_Quantitative_research.backend.core.api import data_api


def test_get_history():
    stock_price = data_api.get_current_stock_price("MSFT")
    print(stock_price)


if __name__ == "__main__":
    test_get_history()
