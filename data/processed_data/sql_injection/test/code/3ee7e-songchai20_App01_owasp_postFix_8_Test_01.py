from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/update', methods=['POST'])
def update():
    # รับข้อมูลจากผู้ใช้
    data = request.form.get('data')

    # ตรวจสอบความถูกต้องของข้อมูล
    if not data or len(data) > 100:
        return 'Invalid data', 400

    # เชื่อมต่อกับฐานข้อมูล SQLite
    <fix/>conn = sqlite3.connect('example08_2.db')</fix>
    cursor = conn.cursor()

    <fix/># ใช้ parameterized queries เพื่อลดความเสี่ยงจาก SQL Injection
    cursor.execute("UPDATE settings SET value = ? WHERE key = 'some_key'", (data,))</fix>
    conn.commit()
    conn.close()

    return 'Data updated successfully!'

if __name__ == '__main__':
    app.run(debug=True)
