import re
import base64
import json
from urllib.parse import urlsplit

import requests
from OpenSSL import crypto
from flask import Flask, request, jsonify, redirect, Response
#from ask_sdk import standard


HOST = '0.0.0.0'
PORT = '7050'
TRUSTED_CERTS_PATH = '/etc/ssl/certs/ca-certificates.crt'

app = Flask(__name__)


def verify_sc_url(url: str) -> bool:
    parsed = urlsplit(url)

    scheme: str = parsed.scheme
    netloc: str = parsed.netloc
    path: str = parsed.path

    try:
        port = parsed.port
    except ValueError:
        port = None

    result = (scheme.lower() == 'https' and
              netloc.lower().split(':')[0] == 's3.amazonaws.com' and
              path[:10] == '/echo.api/' and
              (port == 443 or port is None))

    return result


def extract_certs(certs_txt: str) -> list:
    pattern = r'-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----'
    certs_txt = re.findall(pattern, certs_txt, flags=re.DOTALL)
    certs = [crypto.load_certificate(crypto.FILETYPE_PEM, cert_txt) for cert_txt in certs_txt]
    return certs


# TODO: think of decomposition
def verify_signature(signature_chain_url: str, request_body: bytes, signature: str) -> bool:
    cert_chain_get = requests.get(signature_chain_url)
    cert_chain_txt = cert_chain_get.text
    cert_chain = extract_certs(cert_chain_txt)

    amazon_cert: crypto.X509 = cert_chain.pop(0)

    # verify not expired
    verify_expired = not amazon_cert.has_expired()

    # get subject alternative names
    cert_extentions = [amazon_cert.get_extension(i) for i in range(amazon_cert.get_extension_count())]
    subject_alt_names = ''

    for extention in cert_extentions:
        if 'subjectAltName' in str(extention.get_short_name()):
            subject_alt_names = extention.__str__()
            break

    verify_sans = 'echo-api.amazon.com' in subject_alt_names

    # verify certs chain
    store = crypto.X509Store()

    for cert in cert_chain:
        store.add_cert(cert)

    with open(TRUSTED_CERTS_PATH, 'r') as crt_f:
        trusted_certs_txt = crt_f.read()
        trusted_certs = extract_certs(trusted_certs_txt)
        for cert in trusted_certs:
            store.add_cert(cert)

    store_context = crypto.X509StoreContext(store, amazon_cert)

    try:
        store_context.verify_certificate()
        verify_chain = True
    except crypto.X509StoreContextError as e:
        verify_chain = False
        print(e)


    # verify signature
    try:
        crypto.verify(amazon_cert, signature, request_body, 'sha1')
        verify_signature = True
    except crypto.Error as e:
        verify_signature = False
        print(e)

    result = verify_expired and verify_sans and verify_chain and verify_signature

    return result


@app.route('/', methods=['POST'])
def skill():
    request_body: bytes = request.get_data()
    sc_url = request.headers.get('Signaturecertchainurl')
    signature = base64.b64decode(request.headers.get('Signature'))
    payload = request.get_json()

    if not verify_sc_url(sc_url):
        return jsonify({'error': 'failed signature certificate URL check'}), 400

    if not verify_signature(sc_url, request_body, signature):
        return jsonify({'error': 'failed signature certificate URL check'}), 400

    return jsonify({'error': 'error'}), 400


def main():
    app.run(host=HOST, port=PORT)


if __name__ == '__main__':
    main()
