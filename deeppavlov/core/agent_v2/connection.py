from mongoengine import connect

HOST = 'localhost'
PORT = 27017

state_storage = connect(host=HOST, port=PORT)
