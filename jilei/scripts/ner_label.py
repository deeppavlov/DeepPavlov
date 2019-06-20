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
    brand_pattern = line[2]
    type_pattern = line[3]
    quote = line[0]
    start_b = quote.index(brand_pattern)
    start_t = quote.index(type_pattern)
    span1 = (start_b, start_b + len(brand_pattern))
    span2 = (start_t, start_t + len(type_pattern))
    total_list = split(quote)
    total_list = label(total_list, span1, 'Brand')  # B: Brand
    total_list = label(total_list, span2, 'Type')  # T: Type
    total_list = concat_digit(total_list)
    total_list = clean(total_list)
    return  total_list
