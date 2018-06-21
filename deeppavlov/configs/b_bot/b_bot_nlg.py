import random


def get(action, params):
    if action == 'ASK_SLOT':
        if params['slot'] == 'team' and params['team']:
            return "Соперник по матчу для команды '%s'?" % (params['team'][0])
        elif params['slot'] == 'team' and not params['team']:
            return "На какой матч вы хотите узнать коэффициент?"
        else:
            return "Упс, не знаю как и спросить :("
    elif action=='QUERY_COEFFICIENT':
        return "Коэффициент для матча '%s - %s' %s к %s" % (params['team'][0], params['team'][1], random.randint(1, 10), random.randint(1, 10))
    elif action == 'QUERY FORECAST':
        if len(params['team']) == 0:
            return "Вот топ прогнозов на сегодня: ..."
        elif len(params['team']) == 1:
            return "Вот топ прогнозов для команды '%s': ..." % params['team'][0]
        else:
            return "Вот топ прогнозов для матча '%s - %s': ..." % (params['team'][0], params['team'][1])
    elif action=='RATING':
        return "Вот рейтинг лучших букмекеров: ..."
    else:
        return "Извинете, вы спросили что-то непонятное. Попробуйте спросить по другому."
