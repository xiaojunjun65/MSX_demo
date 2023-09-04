from functools import wraps
from itsdangerous import SignatureExpired, BadSignature
from authlib.jose import jwt, JoseError

import json

from werkzeug.exceptions import HTTPException
from flask import jsonify


class APIException(HTTPException):
    code = 500
    message = 'sorry, we made a mistake (*￣︶￣)!'
    error_code = 239999

    def __init__(self, error_code=None, message=None, code=None):
        if code:
            self.code = code
        if error_code:
            self.error_code = error_code
        if message:
            self.message = message
        super(APIException, self).__init__(message, None)

    def get_body(self, environ=None):
        body = {
            'message': self.message,
            'code': self.error_code
        }
        return json.dumps(body)

    def to_response(self):
        return jsonify({
            'message': self.message,
            'code': self.error_code
        })


class NotFound(APIException):
    code = 404
    message = 'resource not found'
    error_code = None


class AuthError(APIException):
    code = 400
    message = 'valid token required'
    error_code = 230001


class TokenExpired(APIException):
    code = 400
    message = 'token expired'
    error_code = 230002


class TokenError(APIException):
    code = 400
    message = 'token wrong'
    error_code = 230003


class PredictionError(APIException):
    # 此异常返回原始异常信息，无特定的错误码
    code = 400
    error_code = 230301

    def get_body(self, environ=None):
        body = {
            'error': self.message,
        }
        return json.dumps(body)


class ModelStateException(APIException):
    code = 501
    error_code = 230302


def get_user_from_token(api_key, token):
    # s = Serializer(api_key)
    try:
        # data = s.loads(token)
        data = jwt.decode(token, api_key)
    except SignatureExpired:
        raise TokenExpired
    except BadSignature:
        raise TokenError
    except Exception:
        raise AuthError

    return data['user_id']


def get_user_from_request(api_key, request, must_auth=False):
    token = request.headers.get("token")
    if not token:
        if must_auth:
            raise AuthError
        return "Mr.Nobody"
    return get_user_from_token(api_key, token)
