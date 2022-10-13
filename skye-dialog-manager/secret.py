# 3rd-party
from connexion.exceptions import OAuthProblem

TOKEN_DB = {
    'f234cf784e7c9669929122343a808bcf9607e425': { #If you want to test this chatbot api, use this token
        'uid': 'api connect test'
    },
    'becec424c4eeb8510ad7c819b03889e671c01e87': {
        'uid': 'dev taemin lee'
    },
    '9b9a6c8ad0192894c0b2e889db7198c9da46a1ce': {
        'uid': 'dev jaehyung seo'
    },
    '4c4f00a6afd26e7bcd2a83bd9bd03ba600dfb6d5': {
        'uid': 'dev jeongbae park'
    }
}


def apikey_auth(token, required_scopes):
    info = TOKEN_DB.get(token, None)

    if not info:
        raise OAuthProblem('Invalid token')

    return info


def get_secret(user) -> str:
    return {"msg:":"{user}님 반갑습니다.".format(user=user)}