from dataclasses import dataclass

import psycopg2
import pytz

from config import get_connection


@dataclass
class Applicant:
    email: str
    registration_time: pytz

    def on_save(self) -> int:
        connection = get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                <vul/>f"insert into applicants (email, registration_time) values ('{self.email}', '{self.registration_time}');")</vul>
            connection.commit()
        except psycopg2.IntegrityError:
            print("this email already exists")
            return 1
        connection.close()
        return 0
