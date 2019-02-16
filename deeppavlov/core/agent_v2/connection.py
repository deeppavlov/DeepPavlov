from mongoengine import connect

DB_NAME = 'test'
HOST = 'localhost'
PORT = 27017

state_storage = connect(host=HOST, port=PORT, db=DB_NAME)
