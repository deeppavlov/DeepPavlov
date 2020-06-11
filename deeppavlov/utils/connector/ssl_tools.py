# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import re
import ssl
from logging import getLogger
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit

import requests
from OpenSSL import crypto

log = getLogger(__name__)


def verify_sc_url(url: str) -> bool:
    """Verify signature certificate URL against Amazon Alexa requirements.

    Batch of dialog IDs can be provided, in other case utterances indexes in
    incoming batch are used as dialog IDs.

    Args:
        url: Signature certificate URL from SignatureCertChainUrl HTTP header.

    Returns:
        result: True if verification was successful, False if not.
    """
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
              path.startswith('/echo.api/') and
              (port == 443 or port is None))

    return result


def extract_certs(certs_txt: str) -> List[crypto.X509]:
    """Extracts pycrypto X509 objects from SSL certificates chain string.

    Args:
        certs_txt: SSL certificates chain string.

    Returns:
        result: List of pycrypto X509 objects.
    """
    pattern = r'-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----'
    certs_txt = re.findall(pattern, certs_txt, flags=re.DOTALL)
    certs = [crypto.load_certificate(crypto.FILETYPE_PEM, cert_txt) for cert_txt in certs_txt]
    return certs


def verify_sans(amazon_cert: crypto.X509) -> bool:
    """Verifies Subject Alternative Names (SANs) for Amazon certificate.

    Args:
        amazon_cert: Pycrypto X509 Amazon certificate.

    Returns:
        result: True if verification was successful, False if not.
    """
    cert_extentions = [amazon_cert.get_extension(i) for i in range(amazon_cert.get_extension_count())]
    subject_alt_names = ''

    for extention in cert_extentions:
        if 'subjectAltName' in str(extention.get_short_name()):
            subject_alt_names = extention.__str__()
            break

    result = 'echo-api.amazon.com' in subject_alt_names

    return result


def verify_certs_chain(certs_chain: List[crypto.X509], amazon_cert: crypto.X509) -> bool:
    """Verifies if Amazon and additional certificates creates chain of trust to a root CA.

    Args:
        certs_chain: List of pycrypto X509 intermediate certificates from signature chain URL.
        amazon_cert: Pycrypto X509 Amazon certificate.

    Returns:
        result: True if verification was successful, False if not.
    """
    store = crypto.X509Store()

    # add certificates from Amazon provided certs chain
    for cert in certs_chain:
        store.add_cert(cert)

    # add CA certificates
    default_verify_paths = ssl.get_default_verify_paths()

    default_verify_file = default_verify_paths.cafile
    default_verify_file = Path(default_verify_file).resolve() if default_verify_file else None

    default_verify_path = default_verify_paths.capath
    default_verify_path = Path(default_verify_path).resolve() if default_verify_path else None

    ca_files = [ca_file for ca_file in default_verify_path.iterdir()] if default_verify_path else []
    if default_verify_file:
        ca_files.append(default_verify_file)

    for ca_file in ca_files:
        ca_file: Path
        if ca_file.is_file():
            with ca_file.open('r', encoding='ascii') as crt_f:
                ca_certs_txt = crt_f.read()
                ca_certs = extract_certs(ca_certs_txt)
                for cert in ca_certs:
                    store.add_cert(cert)

    # add CA certificates (Windows)
    ssl_context = ssl.create_default_context()
    der_certs = ssl_context.get_ca_certs(binary_form=True)
    pem_certs = '\n'.join([ssl.DER_cert_to_PEM_cert(der_cert) for der_cert in der_certs])
    ca_certs = extract_certs(pem_certs)
    for ca_cert in ca_certs:
        store.add_cert(ca_cert)

    store_context = crypto.X509StoreContext(store, amazon_cert)

    try:
        store_context.verify_certificate()
        result = True
    except crypto.X509StoreContextError:
        result = False

    return result


def verify_signature(amazon_cert: crypto.X509, signature: str, request_body: bytes) -> bool:
    """Verifies Alexa request signature.

    Args:
        amazon_cert: Pycrypto X509 Amazon certificate.
        signature: Base64 decoded Alexa request signature from Signature HTTP header.
        request_body: full HTTPS request body
    Returns:
        result: True if verification was successful, False if not.
    """
    signature = base64.b64decode(signature)

    try:
        crypto.verify(amazon_cert, signature, request_body, 'sha1')
        result = True
    except crypto.Error:
        result = False

    return result


def verify_cert(signature_chain_url: str) -> Optional[crypto.X509]:
    """Conducts series of Alexa SSL certificate verifications against Amazon Alexa requirements.

    Args:
        signature_chain_url: Signature certificate URL from SignatureCertChainUrl HTTP header.
    Returns:
        result: Amazon certificate if verification was successful, None if not.
    """
    try:
        certs_chain_get = requests.get(signature_chain_url)
    except requests.exceptions.ConnectionError as e:
        log.error(f'Amazon signature chain get error: {e}')
        return None

    certs_chain_txt = certs_chain_get.text
    certs_chain = extract_certs(certs_chain_txt)

    amazon_cert: crypto.X509 = certs_chain.pop(0)

    # verify signature chain url
    sc_url_verification = verify_sc_url(signature_chain_url)
    if not sc_url_verification:
        log.error(f'Amazon signature url {signature_chain_url} was not verified')

    # verify not expired
    expired_verification = not amazon_cert.has_expired()
    if not expired_verification:
        log.error(f'Amazon certificate ({signature_chain_url}) expired')

    # verify subject alternative names
    sans_verification = verify_sans(amazon_cert)
    if not sans_verification:
        log.error(f'Subject alternative names verification for ({signature_chain_url}) certificate failed')

    # verify certs chain
    chain_verification = verify_certs_chain(certs_chain, amazon_cert)
    if not chain_verification:
        log.error(f'Certificates chain verification for ({signature_chain_url}) certificate failed')

    result = (sc_url_verification and expired_verification and sans_verification and chain_verification)

    return amazon_cert if result else None
