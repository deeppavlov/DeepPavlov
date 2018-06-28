def get_product_info(s):
    return f"Тарифы на обслуживание для продукта {s['product']} вы можете узнать по ссылке ...", s


def get_service_info(s):
    return f"Тарифы на обслуживание для услуги {s['service']} вы можете узнать по ссылке ...", s


def ask_slot(state, params):
    if params['slot'] == 'product':
        state['__COMMANDS__'].append({'command': 'FILL_SLOT', 'slot': 'product'})
        return "Стоимость какого продукта Вас интересует?", state
    elif params['slot'] == 'service':
        state['__COMMANDS__'].append({'command': 'FILL_SLOT', 'slot': 'service'})
        return "Стоимость какой услуги Вы бы хотели узнать?", state
    else:
        return "Упс, не знаю как и спросить :(", state


def get():
    return [
        (lambda s: s["product"], get_product_info),
        (lambda s: s["service"], get_service_info),
        (lambda s: len(s["intent"]) >=1 and s["intent"][0] == "PRODUCT", lambda s: ask_slot(s, {"slot": "product"})),
        (lambda s: len(s["intent"]) >=1 and s["intent"][0] == "SERVICE", lambda s: ask_slot(s, {"slot": "service"})),
        (lambda s: not s["intent"] and not s["product"] and not s["service"],
         lambda s: ("Уточните стоимость какого продукта или услуги Вы бы хотели узнать?", s)),
        (lambda s: True, lambda s: ("Я вас не понимаю. Спросите, пожалуйста, по другому", s))
    ]