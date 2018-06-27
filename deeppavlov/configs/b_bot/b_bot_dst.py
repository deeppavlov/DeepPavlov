def command_append_to_slot(c, s, p):
    slot = c['slot']
    if p[slot]:
        s[slot] += p[slot]
    return s


def command_update_slot(c, s, p):
    slot = c['slot']
    if p[slot]:
        s[slot] = p[slot]
    return s


def command_default(c, s, p):
    s.clear()
    s.update(p)
    return s


def get():
    return {
        'APPEND_TO_SLOT':  command_append_to_slot,
        'UPDATE_SLOT': command_update_slot,
        'DEFAULT': command_default
    }