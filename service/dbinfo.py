# 创建 MySQL 数据库连接
import mysql.connector


mydb = mysql.connector.connect(
  host="192.168.100.1",
  user="root",
  password="adwdsAD23ddasa",
  database="iisdb"
)