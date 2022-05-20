# Khalikov R.V. 
import pandas as pd
import os

def task1(df_order, df_product):
    """
    1. По какой категории самое высокое проникновение в трафик? Чему оно равно?
    :param df_order:
    :param df_product:
    :return:
    """
    df = df_order.merge(df_product)
    category, count_val = df["level1"].value_counts().index[0], df["level1"].value_counts()[0]

    print(f"1. По категории {category} самое высокое проникновение в трафик. оно равно {count_val}")
    return

def task2(df_order):
    """
    2. Какой средний чек был 13.01?
    :param df_order:
    :param df_product:
    :return:
    """
    df_1301 = df_order.query("'2022-01-13' <= accepted_at < '2022-01-14'").copy()
    df_1301["sum_order"] = df_1301["quantity"].multiply(df_1301["price"])
    df = df_1301[["order_id", "sum_order"]].groupby(by="order_id").agg(sum).reset_index()

    print(f"2. средний чек == {df['sum_order'].mean():0.2f}руб за 13.01")
    return


def task3(df_order, df_product):
    """
    3. Какая доля промо по категории Сыры?
    :param df_order:
    :param df_product:
    :return:
    """
    df = df_order.merge(df_product)
    cols = ["product_id", "regular_price", "price"]
    df = df.loc[df["level1"] == "Сыры", cols].drop_duplicates()
    mask = df["regular_price"] > df["price"]
    count_promo = df[mask].shape[0]
    percent_promo = count_promo / df.shape[0]

    print(f"3. доля промо по категории Сыры == {percent_promo:0.2f} %")
    return


def task4(df_order, df_product):
    """
    4. Чему равно потребление по категории Птица?
    :param df_order:
    :param df_product:
    :return:
    """
    df = df_order.merge(df_product)
    count_bird = df.loc[df["level1"] == "Птица", "quantity"].sum()
    print(f"4. Чему равно потребление по категории Птица? == {count_bird}")

    return

def task5(df_order, df_product):
    """
    5. Какая маржа в руб и % по категории Молочная продукция?
    :param df_order:
    :param df_product:
    :return:
    """
    df = df_order.merge(df_product)
    df = df[df["level1"] == "Молочная продукция"]
    df["sum_cost"] = df["cost_price"] * df["quantity"]
    df["sum_sale"] = df["price"] * df["quantity"]

    margin_sum = df["sum_sale"].sum() - df["sum_cost"].sum()
    margin_percent = margin_sum / df["sum_sale"].sum()

    print(f"5. маржа в руб == {margin_sum} и % == {margin_percent:0.2f} по категории Молочная продукция")
    return

def main():
    print(os.getcwd())
    df_order = pd.read_excel(
        "Тестовое задание (Ведущий аналитик)_.xlsx",
        sheet_name="t_order"
    )

    df_product = pd.read_excel(
        "Тестовое задание (Ведущий аналитик)_.xlsx",
        sheet_name="t_products"
    )

    task1(df_order, df_product)
    task2(df_order)
    task3(df_order, df_product)
    task4(df_order, df_product)
    task5(df_order, df_product)

if __name__ == "__main__":
    main()

