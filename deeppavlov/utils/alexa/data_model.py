from typing import Dict

from fastapi import Body, Header
from pydantic import BaseModel, conlist

example_signature = 'Z5H5wqd06ExFVPNfJiqhKvAFjkf+cTVodOUirucHGcEVAMO1LfvgqWUkZ/X1ITDZbI0w+SMwVkEQZlkeThbVS/54M22StNDUtfz4Ua20xNDpIPwcWIACAmZ38XxbbTEFJI5WwqrbilNcfzqiGrIPfdO5rl+/xUjHFUdcJdUY/QzBxXsceytVYfEiR9MzOCN2m4C0XnpThUavAu159KrLj8AkuzN0JF87iXv+zOEeZRgEuwmsAnJrRUwkJ4yWokEPnSVdjF0D6f6CscfyvRe9nsWShq7/zRTa41meweh+n006zvf58MbzRdXPB22RI4AN0ksWW7hSC8/QLAKQE+lvaw=='
example_signaturecertchainurl = 'https://s3.amazonaws.com/echo.api/echo-api-cert-6-ats.pem'
signature_header = Header(..., example=example_signature, alias='Signature')
cert_chain_url_header = Header(..., example=example_signaturecertchainurl, alias='Signaturecertchainurl')


class Application(BaseModel):
    applicationId: str = Body(..., example='amzn1.ask.skill.8b17a5de-3749-4919-aa1f-e0bbaf8a46a6')


class Attributes(BaseModel):
    sessionId: str = Body(..., example='amzn1.echo-api.session.3c6ebffd-55b9-4e1a-bf3c-c921c1801b63')


class User(BaseModel):
    userId: str = Body(..., example='amzn1.ask.account.AGR4R2LOVHMNMNOGROBVNLU7CL4C57X465XJF2T2F55OUXNTLCXDQP3I55UXZIALEKKZJ6Q2MA5MEFSMZVPEL5NVZS6FZLEU444BVOLPB5WVH5CHYTQAKGD7VFLGPRFZVHHH2NIB4HKNHHGX6HM6S6QDWCKXWOIZL7ONNQSBUCVPMZQKMCYXRG5BA2POYEXFDXRXCGEVDWVSMPQ')


class Session(BaseModel):
    new: bool = Body(..., example=False)
    sessionId: str = Body(..., example='amzn1.echo-api.session.3c6ebffd-55b9-4e1a-bf3c-c921c1801b63')
    application: Application
    attributes: Attributes
    user: User


class Device(BaseModel):
    deviceId: str = Body(..., example='amzn1.ask.device.AFQAMLYOYQUUACSE7HFVYS4ZI2KUB35JPHQRUPKTDCAU3A47WESP5L57KSWT5L6RT3FVXWH4OA2DNPJRMZ2VGEIACF3PJEIDCOUWUBC4W5RPJNUB3ZVT22J4UJN5UL3T2UBP36RVHFJ5P4IPT2HUY3P2YOY33IOU4O33HUAG7R2BUNROEH4T2')
    supportedInterfaces: Dict = Body(..., example={})


class System(BaseModel):
    application: Application
    user: User
    device: Device
    apiEndpoint: str = Body(..., example='https://api.amazonalexa.com')
    apiAccessToken: str = Body(..., example='eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6IjEifQ.eyJhdWQiOiJodHRwczovL2FwaS5hbWF6b25hbGV4YS5jb20iLCJpc3MiOiJBbGV4YVNraWxsS2l0Iiwic3ViIjoiYW16bjEuYXNrLnNraWxsLjhiMTdhNWRlLTM3NDktNDkxOS1hYTFmLWUwYmJhZjhhNDZhNiIsImV4cCI6MTU0NTIyMzY1OCwiaWF0IjoxNTQ1MjIwMDU4LCJuYmYiOjE1NDUyMjAwNTgsInByaXZhdGVDbGFpbXMiOnsiY29uc2VudFRva2VuIjpudWxsLCJkZXZpY2VJZCI6ImFtem4xLmFzay5kZXZpY2UuQUZRQU1MWU9ZUVVVQUNTRTdIRlZZUzRaSTJLVUIzNUpQSFFSVVBLVERDQVUzQTQ3V0VTUDVMNTdLU1dUNUw2UlQzRlZYV0g0T0EyRE5QSlJNWjJWR0VJQUNGM1BKRUlEQ09VV1VCQzRXNVJQSk5VQjNaVlQyMko0VUpONVVMM1QyVUJQMzZSVkhGSjVQNElQVDJIVVkzUDJZT1kzM0lPVTRPMzNIVUFHN1IyQlVOUk9FSDRUMiIsInVzZXJJZCI6ImFtem4xLmFzay5hY2NvdW50LkFHUjRSMkxPVkhNTk1OT0dST0JWTkxVN0NMNEM1N1g0NjVYSkYyVDJGNTVPVVhOVExDWERRUDNJNTVVWFpJQUxFS0taSjZRMk1BNU1FRlNNWlZQRUw1TlZaUzZGWkxFVTQ0NEJWT0xQQjVXVkg1Q0hZVFFBS0dEN1ZGTEdQUkZaVkhISDJOSUI0SEtOSEhHWDZITTZTNlFEV0NLWFdPSVpMN09OTlFTQlVDVlBNWlFLTUNZWFJHNUJBMlBPWUVYRkRYUlhDR0VWRFdWU01QUSJ9fQ.jcomYhBhU485T4uoe2NyhWnL-kZHoPQKpcycFqa-1sy_lSIitfFGup9DKrf2NkN-I9lZ3xwq9llqx9WRN78fVJjN6GLcDhBDH0irPwt3n9_V7_5bfB6KARv5ZG-JKOmZlLBqQbnln0DAJ10D8HNiytMARNEwduMBVDNK0A5z6YxtRcLYYFD2-Ieg_V8Qx90eE2pd2U5xOuIEL0pXfSoiJ8vpxb8BKwaMO47tdE4qhg_k7v8ClwyXg3EMEhZFjixYNqdW1tCrwDGj58IWMXDyzZhIlRMh6uudMOT6scSzcNVD0v42IOTZ3S_X6rG01B7xhUDlZXMqkrCuzOyqctGaPw')


class Experience(BaseModel):
    arcMinuteWidth: int = Body(..., example=246)
    arcMinuteHeight: int = Body(..., example=144)
    canRotate: bool = Body(..., example=False)
    canResize: bool = Body(..., example=False)


class Viewport(BaseModel):
    experiences: conlist(Experience, min_items=1)
    shape: str = Body(..., example='RECTANGLE')
    pixelWidth: int = Body(..., example=1024)
    pixelHeight: int = Body(..., example=600)
    dpi: int = Body(..., example=160)
    currentPixelWidth: int = Body(..., example=1024)
    currentPixelHeight: int = Body(..., example=600)
    touch: conlist(str, min_items=1) = Body(..., example=['SINGLE'])


class Context(BaseModel):
    system: System = Body(..., alias='System')
    viewport: Viewport = Body(..., alias='Viewport')


class Status(BaseModel):
    code: str = Body(..., example='ER_SUCCESS_NO_MATCH')


class RPA(BaseModel):
    authority: str = Body(..., example='amzn1.er-authority.echo-sdk.amzn1.ask.skill.8b17a5de-3749-4919-aa1f-e0bbaf8a46a6.GetInput')
    status: Status


class Resolutions(BaseModel):
    resolutionsPerAuthority: conlist(RPA, min_items=1)


class RawInput(BaseModel):
    name: str = Body(..., example='raw_input')
    value: str = Body(..., example='my beautiful sandbox skill')
    resolutions: Resolutions
    confirmationStatus: str = Body(..., example='NONE')
    source: str = Body(..., example='USER')


class Slots(BaseModel):
    raw_input: RawInput


class Intent(BaseModel):
    name: str = Body(..., example='AskDeepPavlov')
    confirmationStatus: str = Body(..., example='NONE')
    slots: Slots


class Request(BaseModel):
    type: str = Body(..., example='IntentRequest')
    requestId: str = Body(..., example='amzn1.echo-api.request.388d0f6e-04b9-4450-a687-b9abaa73ac6a')
    timestamp: str = Body(..., example='2018-12-19T11:47:38Z')
    locale: str = Body(..., example='en-US')
    intent: Intent


class Data(BaseModel):
    version: str = Body(..., example='1.0')
    session: Session
    context: Context
    request: Request
