"""
Example:
    1. 松下液晶电视TH-50FX700C
        松   B-Brand
        下   I-Brand
        液   0
        晶   0
        电   0
        视   0
        T    B
        H    I-Type
        -    I-Type
        50   I-Type
        F    I-Type
        X    I-Type
        700  I-Type
        C    I-Type
       。    0

"""

import re
import pandas as pd

def split(quote):
    total_list = []
    for chara in quote:
        sub_list = []
        sub_list.append(chara)
        sub_list.append(0)  # 实体标记
        total_list.append(sub_list)
    return total_list

def label(total_list, span, label):
    for i in range(span[0],span[1]):
        total_list[i][1] = 'I-' + label
    total_list[span[0]][1] = 'B-' + label
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
               new_list.append([temp, capture[0][1]])
               capture = []
           new_list.append(total_list[i])
        else:
           capture.append(total_list[i])
    if capture != []:
        temp = ''
        for j in range(len(capture)):
            temp += capture[j][0]
        new_list.append([temp, capture[0][1]])
    new_list.append(['。', 0])
    return new_list

def clean(total_list):
    out_list = []
    for line in total_list:
        if line[0] != ' ':
            out_list.append(line)
    return out_list

def tag(line):
    i,j,k = 2, 3, 1
    brand_pattern = line[i]
    type_pattern = line[j]
    quote = line[k]
    span1 = (quote.index(brand_pattern), quote.index(brand_pattern) + len(brand_pattern))
    span2 = (quote.index(type_pattern), quote.index(type_pattern) + len(type_pattern))
    total_list = split(quote)
    total_list = label(total_list, span1, 'Brand') # B: Brand
    total_list = label(total_list, span2, 'Type') # T: Type
    total_list = concat_digit(total_list)
    total_list = clean(total_list)
    return  total_list

# string = ['彩电','松下液晶电视TH-50FX700C','松下','TH-50FX700C']
# string = ['彩电','Haier/海尔彩电 LQ55S31N','海尔','LQ55S31N']
# string = ['电饭煲','飞利浦智能电饭煲HD4531/00','飞利浦','HD4531/00']
#
# # string = '松下（Panasonic） TH-32E380C 32英寸 窄边框 高清LED 卧室客厅 液晶平板电视'
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
