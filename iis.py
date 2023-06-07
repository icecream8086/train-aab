from aiohttp.web_app import Application
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from service.dbinfo import mydb
from service.api.base import BaseResponse
from service.api.user.login import UserLogin
from service.jwt import encode_token,decode_token

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_secret_key'  # 设置密钥，用于签名
api = Api(app)


        
UserLogin.apptag = app  # 将 app 对象保存为 UserLogin 类的类变量


api.add_resource(UserLogin, '/api/user/login')  # 添加路由


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

