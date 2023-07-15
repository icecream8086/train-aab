from flask import request, jsonify
from flask_restful import Resource
from service.dbinfo import mydb
from service.api.base import BaseResponse
from service.jwt import encode_token

from flask import request, jsonify
from flask_restful import Resource
from service.dbinfo import mydb
from service.api.base import BaseResponse
from service.jwt import encode_token

class UserLogin(BaseResponse, Resource):
    apptag = None

    def __init__(self):
        super().__init__()

    def post(self):
        username = request.form.get('username')
        password = request.form.get('password')
        # 创建 MySQL 数据库游标
        mycursor = mydb.cursor()
        # 执行 SQL 查询
        sql = "SELECT * FROM users WHERE (name = %s OR email = %s) AND password = %s"
        val = (username, username, password)
        mycursor.execute(sql, val)
        # 检查是否有匹配的行
        result = mycursor.fetchone()
        if result:
            payload = {'username': result[1]}
            token = encode_token(payload, self.apptag.config['SECRET_KEY'])
            # 允许继承后修改
            # self.response['message'] = "登录成功"
            self.response['data'] = {"id":result[0],"username": result[1], "accessToken": token}
        else:
            self.response['code'] = 401
            self.response['message'] = "用户名或密码错误"
        # 关闭游标
        mycursor.close()
        return jsonify(self.response)