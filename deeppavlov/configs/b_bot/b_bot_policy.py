import random
from collections import defaultdict
import datetime


def action_ask_slot(state, params):
    if params['slot'] == 'team' and state['team']:
        state['__COMMANDS__'].append({'command': 'APPEND_TO_SLOT', 'slot': 'team'})
        return "Соперник по матчу для команды '%s'?" % (state['team'][0]), state
    elif params['slot'] == 'team' and not state['team']:
        state['__COMMANDS__'].append({'command': 'APPEND_TO_SLOT', 'slot': 'team'})
        return "На какой матч вы хотите узнать коэффициент?", state
    else:
        return "Упс, не знаю как и спросить :(", state


def action_query_coefficient(state, params):
    return "Коэффициент для матча '%s - %s' %s к %s" % (
        params['team'][0], params['team'][1], random.randint(1, 10), random.randint(1, 10)), state


def action_query_forecast(state, params):
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


def action_query_rating(state, params):
    return "Вот рейтинг лучших букмекеров: ...", state


def get():
    return [
        (lambda s: s['coefficient'] and len(s['team']) < 2, lambda s: action_ask_slot(s, {"slot": 'team'})),
        (lambda s: s['coefficient'] and len(s['team']) >= 2, lambda s: action_query_coefficient(s, {k: s[k] for k in ['team', 'market', 'bet', 'bookmaker']})),
        (lambda s: s['forecast'], lambda s: action_query_forecast(s, {k: s[k] for k in ['team', 'dates', 'sporе', 'event', 'expert', 'bookmaker']})),
        (lambda s: (not s['coefficient']) and (s['team'] or s['dates'] or s['sport'] or s['event'] or s['expert'] or s['bookmaker'] or s['dates']),
            lambda s: action_query_forecast(s, {k: s[k] for k in ['team', 'dates', 'sporе', 'event', 'expert', 'bookmaker']})),
        (lambda s: s['rating'], lambda s: action_query_rating(s, {})),
        (lambda s: True, lambda s: ('DEFAULT_ACTION', s))
    ]

