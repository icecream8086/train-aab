import requests

url = 'http://127.0.0.1:5000/api/user/login'  # 请求的 URL
data = {'username': 'admin', 'password': 'password'}  # 请求的数据
response = requests.post(url, data=data)  # 发送 POST 请求

print(response.status_code)  # 打印响应状态码
print(response.json())  # 打印响应内容