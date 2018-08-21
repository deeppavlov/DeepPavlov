def siberia_region(s):
    return f"Для подключения к услуге необходимо предоставить заявление в офис банка по месту открытия счета", s


def south_west_region(s):
    return f"Для подключения к услуге необходимо предоставить заявление в офис банка по месту открытия счета", s


def other_region(s):
    return f"Данная услуга не предоставляется. Вы можете узнать информацию об операциях по счету через систему СББОЛ", s


def ask_region(state, params):
        state['__COMMANDS__'].append({'command': 'FILL_SLOT', 'slot': 'region'})
        return "В каком регионе у вас открыт счет?", state


def get():
    return [
        (lambda s: not s["region"], lambda s: ask_region(s, 'region')),
        (lambda s: s["region"][0] == 'SIBERIA', siberia_region),
        (lambda s: s["region"][0] == 'SOUTH_WEST', south_west_region),
        (lambda s: s["region"][0] == 'OTHER', other_region),
        (lambda s: True, lambda s: ("Я могу рассказать как подключить смс-информирование по расчетному счет. В каком регионе у вас открыт счет?", s))
    ]