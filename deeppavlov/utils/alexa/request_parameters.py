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

"""Request parameters for the DeepPavlov model launched as a skill for Amazon Alexa.

Request parameters from this module are used to declare additional information
and validation for request parameters to the DeepPavlov model launched as
a skill for Amazon Alexa.

See details at https://fastapi.tiangolo.com/tutorial/header-params/,
               https://fastapi.tiangolo.com/tutorial/body-multiple-params/

"""

from fastapi import Header, Body

_signature_example = 'Z5H5wqd06ExFVPNfJiqhKvAFjkf+cTVodOUirucHGcEVAMO1LfvgqWUkZ/X1ITDZbI0w+SMwVkEQZlkeThbVS/54M22StNDUtfz4Ua20xNDpIPwcWIACAmZ38XxbbTEFJI5WwqrbilNcfzqiGrIPfdO5rl+/xUjHFUdcJdUY/QzBxXsceytVYfEiR9MzOCN2m4C0XnpThUavAu159KrLj8AkuzN0JF87iXv+zOEeZRgEuwmsAnJrRUwkJ4yWokEPnSVdjF0D6f6CscfyvRe9nsWShq7/zRTa41meweh+n006zvf58MbzRdXPB22RI4AN0ksWW7hSC8/QLAKQE+lvaw=='
_signature_cert_chain_url_example = 'https://s3.amazonaws.com/echo.api/echo-api-cert-6-ats.pem'
_body_example = {
    "version": "1.0",
    "session": {
        "new": True,
        "sessionId": "amzn1.echo-api.session.ee48c20e-5ad5-461f-a735-ce058491e914",
        "application": {
            "applicationId": "amzn1.ask.skill.52b86ebd-dd7d-45c3-a763-de584f62b8d6"
        },
        "user": {
            "userId": "amzn1.ask.account.AHUAJ5RRTJDATP63AIRLNOVBC2QCJ7U5WSVSD432EA45PDVWAX5CQ6Z2OLD2H2A77VSBQGIMIWAVBMWLHK2EVZAE5VVJ2FHWS4AQM3GMIDH62GZBZ4DOUWXA3DXRBBXXXTKAITDUCZTLG5GP3XN7YORE5FQO2MERGKK7WAJUTHPMLYN4W2IUBVYDIW7544M57N4KV5HMS4DESMY"
        }
    },
    "context": {
        "System": {
            "application": {
                "applicationId": "amzn1.ask.skill.52b86ebd-dd7d-45c3-a763-de584f62b8d6"
            },
            "user": {
                "userId": "amzn1.ask.account.AHUAJ5RRTJDATP63AIRLNOVBC2QCJ7U5WSVSD432EA45PDVWAX5CQ6Z2OLD2H2A77VSBQGIMIWAVBMWLHK2EVZAE5VVJ2FHWS4AQM3GMIDH62GZBZ4DOUWXA3DXRBBXXXTKAITDUCZTLG5GP3XN7YORE5FQO2MERGKK7WAJUTHPMLYN4W2IUBVYDIW7544M57N4KV5HMS4DESMY"
            },
            "device": {
                "deviceId": "amzn1.ask.device.AH777YKPTWMNQGVKUKDWPQOWWEDBDJNMIGP5GHDXOIMI3N5RYZWQ2HBQEOUXMUJEHRBKDX6HCFEA7RRWNAGKHJLSD5KWLTKR35D42TW6BVL64THCYUITTH3G6ZMWZ6GNAELTXWB4YAZJWUK4J2BIFVLUP2KHZNTQRJRBEFGNWY4V2RCEEQOZC",
                "supportedInterfaces": {}
            },
            "apiEndpoint": "https://api.amazonalexa.com",
            "apiAccessToken": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6IjEifQ.eyJhdWQiOiJodHRwczovL2FwaS5hbWF6b25hbGV4YS5jb20iLCJpc3MiOiJBbGV4YVNraWxsS2l0Iiwic3ViIjoiYW16bjEuYXNrLnNraWxsLjUyYjg2ZWJkLWRkN2QtNDVjMy1hNzYzLWRlNTg0ZjYyYjhkNiIsImV4cCI6MTU2OTgzNTY5MiwiaWF0IjoxNTY5ODM1MzkyLCJuYmYiOjE1Njk4MzUzOTIsInByaXZhdGVDbGFpbXMiOnsiY29udGV4dCI6IkFBQUFBQUFBQUFCTm5aUTd4b09EcGNYL0tuMDFpZ1F6S2dFQUFBQUFBQUJSazluemRVNTlQZWVFY0t5SERSZEwzRiszdnZrVGpQWWQ3MnhFYzFQcUNSeStTTWZmaFFscUh4azJuTHNTV01JKzFnZEtYc0t1RGVSQkJqNERTck5TUWVCZjNkbmtxNERWMXRqVjhmUnB1UWRXdlY2bERZN3YycXMyZVRlZEN6V0RLY21oRXFjRHdBNWlmdUxEdzB5bmZVVVh6Rk0yLzBBeDdGUmYxaS9FWXJRaWV0T2Q1dWllYU9RUFUrUUNMUUNRMFI0Ni9Ld1d1SWdxcE5sSGw0bU0xSHNhYXJOS3VzM0hDRzNyNm9LekxkT25EVUFKTDRtajkzSGwwZUhUQ1M0WDFySEtTTHNMNUlxa2hnUTk3a0R0WVovK1dNbkVDNklGUEZ6OHdYYU9jaDJYS05EUTNERVlGWTE0WHRkTXY0MlBYeTJlQ3VjQy9udnU2ZGMxaGRjUGdkZUp2Rmw3WlBBK0RSa2RqYXovL1NNTjVQMlNBY0NqK2JBZXIrTGZOTDByYUxhbGh5OEhleGl5IiwiY29uc2VudFRva2VuIjpudWxsLCJkZXZpY2VJZCI6ImFtem4xLmFzay5kZXZpY2UuQUg3NzdZS1BUV01OUUdWS1VLRFdQUU9XV0VEQkRKTk1JR1A1R0hEWE9JTUkzTjVSWVpXUTJIQlFFT1VYTVVKRUhSQktEWDZIQ0ZFQTdSUldOQUdLSEpMU0Q1S1dMVEtSMzVENDJUVzZCVkw2NFRIQ1lVSVRUSDNHNlpNV1o2R05BRUxUWFdCNFlBWkpXVUs0SjJCSUZWTFVQMktIWk5UUVJKUkJFRkdOV1k0VjJSQ0VFUU9aQyIsInVzZXJJZCI6ImFtem4xLmFzay5hY2NvdW50LkFIVUFKNVJSVEpEQVRQNjNBSVJMTk9WQkMyUUNKN1U1V1NWU0Q0MzJFQTQ1UERWV0FYNUNRNloyT0xEMkgyQTc3VlNCUUdJTUlXQVZCTVdMSEsyRVZaQUU1VlZKMkZIV1M0QVFNM0dNSURINjJHWkJaNERPVVdYQTNEWFJCQlhYWFRLQUlURFVDWlRMRzVHUDNYTjdZT1JFNUZRTzJNRVJHS0s3V0FKVVRIUE1MWU40VzJJVUJWWURJVzc1NDRNNTdONEtWNUhNUzRERVNNWSJ9fQ.brF2UpwjKMbYhR50WdoALbz0CM9hFtfAUw4Hh9-tOMJY8imui3oadv5S6QbQlfYD4_V_mJG2WOfkLmvirdRwdY6gI289WB48a6pK29VVcJWhYv1wIEpNQUMvMQqMZpjUuCI6DR9PqSeHulqPt14ytiA1ghOVSsAsHFXGbhNNeM9SdS1Ss0JQolSvXo09qC3JFRpDBI1bzBxRthhWEwgIEkC-JuFAbCbXz-710FkI4vzlMElgvC2GIsPf-5RaTJXps4UuG1rLieerirrrZfbpmhO0x2vDbLvBCCbqUtoHPyKofexfBXebvMjjJ7PRZvKYxAg3SBVZLvpGVl0prgJ8PA"
        },
        "Viewport": {
            "experiences": [
                {
                    "arcMinuteWidth": 246,
                    "arcMinuteHeight": 144,
                    "canRotate": False,
                    "canResize": False
                }
            ],
            "shape": "RECTANGLE",
            "pixelWidth": 1024,
            "pixelHeight": 600,
            "dpi": 160,
            "currentPixelWidth": 1024,
            "currentPixelHeight": 600,
            "touch": [
                "SINGLE"
            ],
            "video": {
                "codecs": [
                    "H_264_42",
                    "H_264_41"
                ]
            }
        }
    },
    "request": {
        "type": "LaunchRequest",
        "requestId": "amzn1.echo-api.request.9b112eb9-eb11-433d-b6b3-8dba7eab9637",
        "timestamp": "2019-09-30T09:23:12Z",
        "locale": "en-US",
        "shouldLinkResultBeReturned": False
    }
}

signature_header = Header(..., example=_signature_example, alias='Signature')
cert_chain_url_header = Header(..., example=_signature_cert_chain_url_example, alias='Signaturecertchainurl')
data_body = Body(..., example=_body_example)
