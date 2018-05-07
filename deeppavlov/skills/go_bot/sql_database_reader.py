import sqlite3
import re
import os


class DataBase(object):
    def __init__(self, database, req_sub_dict):
        database = os.path.join(os.getcwd(), database)
        print(database)
        if not os.path.isfile(database):
            print('Database does not exist')
            raise IOError

        print(f'Path of the database is {database}')
        # connect to database
        conn = sqlite3.connect(database)
        self.cur = conn.cursor()
        self.req_sub_dict = req_sub_dict

    def sub_reqs(self, string, context_details):
        # searches through entity placeholders in text and substitutes relevant context to the conversation
        for requirement in self.req_sub_dict.keys():
            if requirement in string:
                assert requirement in context_details.keys()
                exec(self.req_sub_dict[requirement][0], locals())
                exec(self.req_sub_dict[requirement][1], locals())
                string = re.sub(requirement, locals()['sub_data'], string)
        return string

    def query(self, query, answer, context_details: dict):
        cur = self.cur
        globals()['cur'] = locals()['cur']
        try:
            query = self.sub_reqs(query, context_details)
        except AssertionError:
            return 'sorry, there is no information for that.'
        exec(query, globals())
        try:
            exec(answer, globals())
            answer_text = globals()['answer_text']
        except IndexError:
            answer_text = 'sorry, there is no information for that.'
        return answer_text
