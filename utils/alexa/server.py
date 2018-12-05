from urllib.parse import urlsplit

import requests
import OpenSSL
from flask import Flask, request, jsonify, redirect, Response
# from ask_sdk

app = Flask(__name__)
HOST = '0.0.0.0'
PORT = '7050'


def verify_sc_url(url: str) -> bool:
    result = True
    parsed = urlsplit(url)

    scheme: str = parsed.scheme
    netloc: str = parsed.netloc
    path: str = parsed.path

    try:
        port = parsed.port
    except ValueError:
        port = None

    result = result and scheme.lower() == 'https'
    result = result and netloc.lower().split(':')[0] == 's3.amazonaws.com'
    result = result and path[:10] == '/echo.api/'
    result = result and (port == 443 or port is None)

    return result


@app.route('/', methods=['POST'])
def skill():
    request_body: bytes = request.get_data()
    print(type(request_body))
    sc_url = request.headers.get('Signaturecertchainurl')

    if not verify_sc_url(sc_url):
        return jsonify({'error': 'failed signature certificate URL check'}), 400

    input=request.get_json()
    print(str(input))
    return None


def main():
    app.run(host=HOST, port=PORT)


if __name__ == '__main__':
    main()
    #print(verify_sc_url('https://s3.amazonaws.com:443/echo.api/echo-api-cert-6-ats.pem'))

