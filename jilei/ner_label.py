"""
Example:
    1. 松下液晶电视TH-50FX700C
        松 B
        下 I
        液 0
        晶 0
        电 0
        视 0
        T  B
        H  I
        -  I
        5  I
        0  I
        F  I
        X  I
        7  I
        0  I
        0  I
        C  I
       。  0

"""

import re

def concat_line(path):
    pattern = ''
    with open(path) as f:
        for line in f:
            line = line.replace('(','\(')\
                       .replace(')','\)')\
                       .replace('+','\+')\
                       .replace('*','\*')\
                       .replace('.','\.')
            pattern += '|' + line.strip('\n')
        pattern = '(' + pattern.strip('|') + ')'
        pattern = pattern.strip('\n')
    return pattern

def get_type():
    path = './type_code.txt'
    pattern = concat_line(path)
    return pattern

def get_brand():
    path = './brand.txt'
    pattern = concat_line(path)
    return pattern

def split(quote):
    total_list = []
    for chara in quote:
        sub_list = []
        sub_list.append(chara)
        sub_list.append(0)  # 实体标记
        sub_list.append(0)  # 类别标记
        total_list.append(sub_list)
    # total_list.append(['。',0,0])
    return total_list

def label(total_list, span, label):
    for i in range(span[0],span[1]):
        total_list[i][1] = 'I'
        total_list[i][2] = label
    total_list[span[0]][1] = 'B'
    # print(total_list[span[0]], span[0])
    return total_list

def concat_digit(total_list):
    new_list = []
    capture = []
    for i in range(len(total_list)):
        if re.match('[0-9]',total_list[i][0]) == None:
           if capture != []:
               temp = ''
               for j in range(len(capture)):
                  temp += capture[j][0]
               new_list.append([temp, capture[0][1], capture[0][2]])
               capture = []

           new_list.append(total_list[i])
        else:
           capture.append(total_list[i])
    new_list.pop()
    new_list.append(['。', 0, 0])
    # print('lenth', len(new_list))
    return new_list

def clean(total_list):
    out_list = []
    for line in total_list:
        if line[0] != ' ':
            out_list.append(line)
    return out_list

def tag(quote):
    brand_pattern = re.compile(get_brand())
    type_pattern = re.compile(get_type())
    span1 = re.search(brand_pattern, quote).span()
    span2 = re.search(type_pattern, quote).span()
    # print(span1)
    # print(span2)
    # span1 = (0,1)
    # span2 = (6,16)
    total_list = split(quote)
    total_list = label(total_list, span1, 'B') # B: Brand
    total_list = label(total_list, span2, 'T') # T: Type
    total_list = concat_digit(total_list)
    total_list = clean(total_list)
    return  total_list

# string = '松下（Panasonic） TH-32E380C 32英寸 窄边框 高清LED 卧室客厅 液晶平板电视'
# out = tag(string)
# for line in out:
#     print(line)

# while True:
#     string = input('type quote: ')
#     out = tag(string)
#     for line in out:
#         print(line)

# out = split('松下液晶电视TH-50FX700C')
# span = (0,1)
# out = label(out, span, 'Brand')
# span = (6,16)
# out = label(out, span, 'Type')
# out = concat_digit(out)
# print(out)
# 松下 (Panasonic)TH-55FX680C 55英寸4k超高清智能wifi网络电视机 辉耀HDR 运动补偿技术
