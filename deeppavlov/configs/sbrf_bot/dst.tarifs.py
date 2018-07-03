def dst_fill_slot(c, s, p):
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
        'FILL_SLOT': dst_fill_slot,
        'DEFAULT': dst_default
    }