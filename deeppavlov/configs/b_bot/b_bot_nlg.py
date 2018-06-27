import random
from collections import defaultdict
import datetime


def get(action, params, state):
    if action == 'ASK_SLOT':
        if params['slot'] == 'team' and params['team']:
            state['COMMANDS'].append({'command': 'APPEND_TO_SLOT', 'slot': 'team'})
            return "Соперник по матчу для команды '%s'?" % (params['team'][0]), state
        elif params['slot'] == 'team' and not params['team']:
            state['COMMANDS'].append({'command': 'APPEND_TO_SLOT', 'slot': 'team'})
            return "На какой матч вы хотите узнать коэффициент?", state
        else:
            return "Упс, не знаю как и спросить :(", state
    elif action=='QUERY_COEFFICIENT':
        return "Коэффициент для матча '%s - %s' %s к %s" % (params['team'][0], params['team'][1], random.randint(1, 10), random.randint(1, 10)), state
    elif action == 'QUERY FORECAST':
        now = datetime.datetime.now()
        d = defaultdict(list)
        if params['dates']:
            d.update(params['dates'][0])
        d['day'] = d['day'] if d['day'] else now.day
        d['month'] = d['month'] if d['month'] else now.month
        d['year'] = d['year'] if d['year'] else now.year
        date = f"{d['day']}/{d['month']}/{d['year']}" if params['dates'] else 'сегодня'
        if len(params['team']) == 0:
            return f"Вот топ прогнозов на {date}: ...", state
        elif len(params['team']) == 1:
            return f"Вот топ прогнозов для команды '%s' на {date}: ..." % params['team'][0], state
        else:
            return f"Вот топ прогнозов для матча '%s - %s' на {date}: ..." % (params['team'][0], params['team'][1]), state
    elif action=='RATING':
        return "Вот рейтинг лучших букмекеров: ...", state
    else:
        return "Извинете, вы спросили что-то непонятное. Попробуйте спросить по другому.", state
