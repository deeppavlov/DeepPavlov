from urllib.parse import urlsplit

import requests
from OpenSSL import crypto
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


def verify_signature(signature_chain_url: str, request_body: bytes) -> bool:
    result = True

    cert_chain_get = requests.get(signature_chain_url)
    cert_chain_txt = cert_chain_get.text
    cert_chain = crypto.load_certificate(crypto.FILETYPE_PEM, cert_chain_txt)

    # verify not expired
    verify_expired = cert_chain.has_expired()

    # get subject alternative names
    cert_extentions = [cert_chain.get_extension(i) for i in range(cert_chain.get_extension_count())]
    subject_alt_names = ''
    for extention in cert_extentions:
        if 'subjectAltName' in str(extention.get_short_name()):
            subject_alt_names = extention.__str__()
            break

    print(subject_alt_names)



    return False


@app.route('/', methods=['POST'])
def skill():
    request_body: bytes = request.get_data()
    sc_url = request.headers.get('Signaturecertchainurl')
    input = request.get_json()

    if not verify_sc_url(sc_url):
        return jsonify({'error': 'failed signature certificate URL check'}), 400

    if not verify_signature(sc_url, request_body):
        return jsonify({'error': 'failed signature certificate URL check'}), 400

    return jsonify({'error': 'error'}), 400


def main():
    app.run(host=HOST, port=PORT)


if __name__ == '__main__':
    main()
    #print(verify_sc_url('https://s3.amazonaws.com:443/echo.api/echo-api-cert-6-ats.pem'))
