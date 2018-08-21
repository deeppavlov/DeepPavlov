def dst_append_to_slot(c, s, p):
    slot = c['slot']
    if p[slot]:
        s[slot] += p[slot]
    return s


def dst_update_slot(c, s, p):
    slot = c['slot']
    if p[slot]:
        s[slot] = p[slot]
    return s


def dst_default(c, s, p):
    s.clear()
    s.update(p)
    return s


def get():
    return {
        'APPEND_TO_SLOT':  dst_append_to_slot,
        'UPDATE_SLOT': dst_update_slot,
        'DEFAULT': dst_default
    }