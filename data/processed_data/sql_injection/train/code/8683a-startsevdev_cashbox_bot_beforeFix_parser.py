import re
from db import Database
from csv_generator import CSVGenerator


class Parser(object):
    def __init__(self):
        self.db = Database("/Users/alexander/code/bots/CashboxBot/data.db")
        self.csv_generator = CSVGenerator()

    def parse_message(self, string):
        string = string.lower()
        date = self.search_date(string)

        if "выручка" in string:
            return self.parse_revenue_message(string, date)
        elif "отчет" in string or "отчёт" in string:
            return self.parse_report_message(string, date)
        # sql-injection protection
        elif string == "'":
            return "Ошибка! Такого товара нет в меню. Попробуй еще раз."
        else:
            return self.db.add_sale(string.capitalize())

    def search_date(self, string):
        search_date_result = re.search("\d{2}.\d{2}.\d{4}", string)
        if search_date_result:
            date = search_date_result.group()
            return date

    def parse_revenue_message(self, string, date):
        if date and self.db.check_date(date):
            return self.db.revenue(date)
        elif date:
            return "Нет данных за этот день."
        elif string == "выручка":
            return self.db.revenue()
        else:
            return "Ошибка! Неверный формат даты."

    def parse_report_message(self, string, date):
        if date and self.db.check_date(date):
            return self.csv_generator.write_csv(date)
        elif date:
            return "Нет данных за этот день."
        elif string == "отчет" or string == "отчёт":
            return self.csv_generator.write_csv()
        else:
            return "Ошибка! Неверный формат даты."
