import sqlite3
import re
import os
import json


class DataBase(object):
    def __init__(self, database, req_sub_dict):
        database = os.path.join(os.getcwd(), database)
        print(database)
        if not os.path.isfile(database):
            print('Database does not exist')
            raise IOError

        # print(f'Path of the database is {database}')
        # connect to database
        conn = sqlite3.connect(database)
        self.cur = conn.cursor()
        self.req_sub_dict = req_sub_dict
        self.drugs = None
        if os.path.isfile('deeppavlov/skills/go_bot/drugs.json'):
            with open('deeppavlov/skills/go_bot/drugs.json', 'r') as f:
                self.drugs = json.load(f)

    def normalize(self, context_details):
        if self.drugs is None:
            return
        if '<drug_name>' in context_details.keys():
            name = context_details['<drug_name>']
            if name not in self.drugs:
                if name.upper() in self.drugs:
                    name = name.upper()
                    context_details['<drug_name>'] = name
                elif name.title() in self.drugs:
                    name = name.title()
                    context_details['<drug_name>'] = name
                else:
                    names = name.split('-')
                    names = [s.title() for s in names]
                    name = '-'.join(names)
                    context_details['<drug_name>'] = name
        if '<condition>' in context_details.keys():
            condition = context_details['<condition>']
            if condition == 'ibd':
                context_details['<condition>'] = 'IBD'
            elif condition == 'tb':
                context_details['<condition>'] = 'TB'

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
        self.normalize(context_details)
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
            print(query)
            answer_text = 'sorry, there is no information for that.'
        return answer_text
