import re
import base64
from pathlib import Path
from typing import List
from urllib.parse import urlsplit

import requests
from OpenSSL import crypto

from deeppavlov.core.common.log import get_logger

ROOT_CERTS_PATH = '/etc/ssl/certs/ca-certificates.crt'

log = get_logger(__name__)


def verify_sc_url(url: str) -> bool:
    """
    Verify signature certificate URL against Amazon Alexa requirements.

    Each call of Agent passes incoming utterances batch through skills filter,
    agent skills, skills processor. Batch of dialog IDs can be provided, in
    other case utterances indexes in incoming batch are used as dialog IDs.

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
              path[:10] == '/echo.api/' and
              (port == 443 or port is None))

    return result


def extract_certs(certs_txt: str) -> List[crypto.X509]:
    """
    Extracts pycrypto X509 objects from SSL certificates chain string.

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
    """
    Verifies Subject Alternative Names (SANs) for Amazon certificate.

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
    """
    Verifies if Amazon and additional certificates creates chain of trust to a root CA.

    Args:
        certs_chain: List of pycrypto X509 intermediate certificates from signature chain URL.
        amazon_cert: Pycrypto X509 Amazon certificate.

    Returns:
        result: True if verification was successful, False if not.
    """
    store = crypto.X509Store()

    for cert in certs_chain:
        store.add_cert(cert)

    root_certs_path = Path(ROOT_CERTS_PATH).resolve()
    with open(root_certs_path, 'r') as crt_f:
        root_certs_txt = crt_f.read()
        root_certs = extract_certs(root_certs_txt)
        for cert in root_certs:
            store.add_cert(cert)

    store_context = crypto.X509StoreContext(store, amazon_cert)

    try:
        store_context.verify_certificate()
        result = True
    except crypto.X509StoreContextError:
        result = False

    return result


def verify_signature(amazon_cert: crypto.X509, signature: str, request_body: bytes) -> bool:
    """
    Verifies Alexa request signature.

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


def verify_cert(signature_chain_url: str, signature: str, request_body: bytes) -> bool:
    """
    Conducts series of Alexa request verifications against Amazon Alexa requirements.

    Args:
        signature_chain_url: Signature certificate URL from SignatureCertChainUrl HTTP header.
        signature: Base64 decoded Alexa request signature from Signature HTTP header.
        request_body: full HTTPS request body
    Returns:
        result: True if verification was successful, False if not.
    """
    certs_chain_get = requests.get(signature_chain_url)
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

    # verify signature
    signature_verification = verify_signature(amazon_cert, signature, request_body)
    if not signature_verification:
        log.error(f'Signature verification for ({signature_chain_url}) certificate failed')

    result = (sc_url_verification and expired_verification and sans_verification
              and chain_verification and signature_verification)

    return result
