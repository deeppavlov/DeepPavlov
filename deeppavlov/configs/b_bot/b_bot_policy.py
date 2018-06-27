def get():
    return [
        (lambda s: bool(s['coefficient']) and len(s['team']) == 0, lambda s: ('ASK_SLOT', {"slot": "team", "team": [], "state": s})),
        (lambda s: bool(s['coefficient']) and len(s['team']) == 1, lambda s: ('ASK_SLOT', {"slot": "team", "team": s['team'], "state": s})),
        (lambda s: bool(s['coefficient']) and len(s['team']) >= 2, lambda s: ('QUERY_COEFFICIENT', {k: s[k] for k in ['team', 'market', 'bet', 'bookmaker']})),
        (lambda s: bool(s['forecast']), lambda s: ('QUERY FORECAST', {k: s[k] for k in ['team', 'date', 'spor', 'event', 'expert', 'bookmaker']})),
        (lambda s: (not bool(s['coefficient'])) and (bool(s['team']) or
                                                     bool(s['date']) or
                                                     bool(s['sport']) or
                                                     bool(s['event']) or
                                                     bool(s['expert']) or
                                                     bool(s['bookmaker']) or
                                                     bool(s['dates'])),
            lambda s: ('QUERY FORECAST', {k: s[k] for k in ['team', 'date', 'spor', 'event', 'expert', 'bookmaker']})),
        (lambda s: bool(s['rating']), lambda s: ('RATING', )),
        (lambda s: True, lambda s: ('DEFAULT_ACTION', s))
    ]