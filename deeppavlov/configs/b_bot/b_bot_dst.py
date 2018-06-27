def update_state(s, p):
    if not s['COMMANDS']:
        s.clear()
        s.update(p)
    else:
        for c in s['COMMANDS']:
            if c['command'] == 'APPEND_TO_SLOT':
                slot = c['slot']
                if p[slot]:
                    s[slot] += p[slot]
            if c['command'] == 'UPDATE_SLOT':
                slot = c['slot']
                if p[slot]:
                    s[slot] = p[slot]
        del s['COMMANDS']
    return s


def get():
    return [
        update_state
    ]