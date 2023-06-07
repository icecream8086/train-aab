from flask_restful import Resource

class BaseResponse(Resource):
    def __init__(self):
        self.response = {
            "code": 200,
            "message": "操作成功",
            "data": {}
        }