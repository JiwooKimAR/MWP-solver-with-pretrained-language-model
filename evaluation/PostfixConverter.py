#@title PostfixConverter 클래스 - 각자 자기가 만든 OP를 여기 합치고 에러가 생기지 않는지 확인해주십시오.

import ast
import itertools
import sys
import math
from contextlib import redirect_stdout
from io import StringIO
from string import ascii_lowercase

from functools import wraps
import errno
import os
import signal

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

class StacknNames: # 2개의 stack으로 구성된 class.
    def __init__(self):
        self.content_stack = [] # 값을 저장하는 stack
        self.name_stack = [] # 변수명을 저장하는 stack
    def pop(self):
        if len(self.content_stack) < 1:
            return None
        return self.content_stack.pop(), self.name_stack.pop()

    def push(self, item, new_name):
        self.content_stack.append(item)
        self.name_stack.append(new_name) # stack에 들어갈땐 항상 새로운 변수명으로 저장됨.

    def size(self):
        return len(self.content_stack)


class PostfixConverter():
    def __init__(self):
        # operator_dic = {operator:['what it does', '# of input', 'type of operator', '# of operand created', '# of lists created']}
        self.operator_dic = {
            '[OP_ADD]':['+', 2, 'infix', 1, 0],                         
            '[OP_SUB]':['-', 2, 'infix', 1, 0], 
            '[OP_MUL]':['*', 2, 'infix', 1, 0],              
            '[OP_DIV]':['/', 2, 'infix', 1, 0],        
            '[OP_FDIV]':['//', 2, 'infix', 1, 0],     
            '[OP_MOD]':['%', 2, 'infix', 1, 0],  
            '[OP_POW]':['**', 2, 'infix', 1, 0],
            '[OP_REVERSE_NUM]':['', 1, 'function', 1, 0],
            '[OP_ABS]':['abs', 1, 'function', 1, 0],
            '[OP_CEIL]':['int', 2, 'function', 1, 0],
            '[OP_FLOOR]':['int', 2, 'function', 1, 0],
            '[OP_ROUND]':['round', 2, 'function', 1, 0],
            '[OP_COMB]':['math.comb', 2, 'function', 1, 0],
            '[OP_PERM]':['math.perm', 2, 'function', 1, 0],
            '[OP_GET_NTH_DECIMAL]':['',2,'function', 1, 0],
            '[OP_DIGIT_UNK_SOLVER]':['',2,'function_special',1, 0],
            '[OP_NUM_UNK_SOLVER]':['',2,'function_special',1, 0],
            '[OP_GEN_POSSIBLE_LIST]':['',1,'function_special'],
            '[OP_GCD]': ['math.gcd', 2, 'function'],
            '[OP_LCM]': ['', 2, 'function'],
            '[OP_DUP_COMB]':['', 2, 'function'],
            '[OP_DAY_DIFF]':['', 2, 'function'],
            '[OP_GET_PI]': ['math.pi', 0, 'function', 1, 0],
            '[OP_LIST_SOL]':['',0,'list', 1, 0],
            '[OP_LIST_EOL]':['',0,'list', 99, 1],
            '[OP_LIST_POP]':['',0,'list', 0 , -1],
            '[OP_LIST_PRIME]':['get_prime', 0, 'list_function', 0, 1],
            '[OP_LIST_DISTINCT]':['', 1, 'list_function'],
            '[OP_LIST_SUM]':['sum',1,'list_function'],
            '[OP_LIST_MEAN]':['average',1,'list_function'],
            '[OP_LIST2NUM]':['',1,'list_function'],
            '[OP_NUM2LIST]':['', 1, 'list_function'],
            '[OP_LIST_INV]':['',1,'list_function'],
            '[OP_LIST_GET_DIVISOR]': ['', 1, 'list_function'],
            '[OP_LIST_NUM2SUM]':['',1,'list_function',0,1],
            '[OP_LIST_LEN]':['len',1,'list_function'],
            '[OP_LIST_GET]':['', 2, 'list_function'],
            '[OP_LIST_INDEX]':['', 2, 'list_function'],
            '[OP_LIST_GET_PERM]':['', 2, 'list_function'],
            '[OP_LIST_GET_PRODUCT]':['', 2, 'list_function'],
            '[OP_LIST_FINDSEQ]':['', 2, 'list_function'],         # 수열과 index를 input으로 받았을 때 수열의 종류(등차or등비)를 찾고 index 번째 (list[index-1]) element를 구해주는 내부연산.
            '[OP_LIST_MORE]':['', 2, 'list_function'],            # 수열과 n를 input으로 받고, n 초과 element들의 sublist를 return하는 연산.
            '[OP_LIST_LESS]':['', 2, 'list_function'],            # 수열과 n를 input으로 받고, n 미만 element들의 sublist를 return하는 연산.
            '[OP_LIST_MORE_EQUAL]':['', 2, 'list_function'],      # 수열과 n를 input으로 받고, n 이상 element들의 sublist를 return하는 연산.
            '[OP_LIST_LESS_EQUAL]':['', 2, 'list_function'],      # 수열과 n를 input으로 받고, n 이하 element들의 sublist를 return하는 연산.
            '[OP_LIST_MAX]':['max',2,'list_function'],            # 수열과 n(int)를 input으로 받고, n번째로 큰 값을 return하고 list는 다시 push하는 연산.
            '[OP_LIST_MIN]':['min',2,'list_function'],            # 수열과 n(int)를 input으로 받고, n번째로 작은 값을 return하고 list는 다시 push하는 연산.
            '[OP_LIST_DIVISIBLE]':['',2,'list_function'],
            '[OP_LIST_FIND_NUM]':['',2,'list_function'],
            '[OP_LIST_FIND_UNK]':['',3,'list_function'],
            '[OP_LIST_DIVIDE_AND_REMAIN]':['',3,'list_function',0,1],
            '[OP_LIST_SEARCH_FIXED_DIGIT]':['',3,'list_function',0,1],
            '[OP_SET_UNION]':['', 2, 'list_function'],
            '[OP_SET_INTERSECT]':['', 2, 'list_function'],
            '[OP_SET_DIFFERENCE]':['', 2, 'list_function'],
            '[OP_LIST_ARANGE]':['', 3, 'list_function'],
            '[OP_LIST_ODD]':['', 2, 'list_function'],
            '[OP_LIST_EVEN]':['', 2, 'list_function'],
            '[OP_LIST_ADD]':['', 2, 'list_function'],
            '[OP_LIST_SUB]':['', 2, 'list_function'],
            '[OP_LIST_MUL]':['', 2, 'list_function'],
            '[OP_LIST_DIV]':['', 2, 'list_function'],
            '[OP_LIST_LEN_MOD_GET]':['', 2, 'list_function'],
            '[OP_LIST_SCALAR_ADD]':['', 2, 'list_function'],
            '[OP_LIST_SCALAR_MUL]':['', 2, 'list_function'],
            '[OP_LIST_RANDOMGAME]':['', 1, 'list_function'],
            '[OP_LIST_COND_BIG_SMALL]':['comp', 3, 'list_function'],
            '[OP_LIST_COND_MAX_MIN]':['comp',2,'list_function'],
            '[OP_MEM]': ['mem', 2, 'aux'],
        }

        self.operand_names = ['var_'+c for c in ascii_lowercase]
        self.list_names = ['list_'+c for c in ascii_lowercase]

        self.operand_stack = StacknNames()
        self.list_stack = StacknNames()
        self.code_string = ''

        self.mem = {}  # {"x": (var_name, value), "y": (var_name, value), ...}

        # convert number to int type if possible, or make it as float type
        self.intifint = lambda x: int(x) if int(x) == self.to_float(x) else self.to_float(x)
    
        # first 1000 prime numbers
        self.prime_num = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999,2003,2011,2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,2237,2239,2243,2251,2267,2269,2273,2281,2287,2293,2297,2309,2311,2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,2657,2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741,2749,2753,2767,2777,2789,2791,2797,2801,2803,2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,2927,2939,2953,2957,2963,2969,2971,2999,3001,3011,3019,3023,3037,3041,3049,3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331,3343,3347,3359,3361,3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,3581,3583,3593,3607,3613,3617,3623,3631,3637,3643,3659,3671,3673,3677,3691,3697,3701,3709,3719,3727,3733,3739,3761,3767,3769,3779,3793,3797,3803,3821,3823,3833,3847,3851,3853,3863,3877,3881,3889,3907,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989,4001,4003,4007,4013,4019,4021,4027,4049,4051,4057,4073,4079,4091,4093,4099,4111,4127,4129,4133,4139,4153,4157,4159,4177,4201,4211,4217,4219,4229,4231,4241,4243,4253,4259,4261,4271,4273,4283,4289,4297,4327,4337,4339,4349,4357,4363,4373,4391,4397,4409,4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,4759,4783,4787,4789,4793,4799,4801,4813,4817,4831,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937,4943,4951,4957,4967,4969,4973,4987,4993,4999,5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851,5857,5861,5867,5869,5879,5881,5897,5903,5923,5927,5939,5953,5981,5987,6007,6011,6029,6037,6043,6047,6053,6067,6073,6079,6089,6091,6101,6113,6121,6131,6133,6143,6151,6163,6173,6197,6199,6203,6211,6217,6221,6229,6247,6257,6263,6269,6271,6277,6287,6299,6301,6311,6317,6323,6329,6337,6343,6353,6359,6361,6367,6373,6379,6389,6397,6421,6427,6449,6451,6469,6473,6481,6491,6521,6529,6547,6551,6553,6563,6569,6571,6577,6581,6599,6607,6619,6637,6653,6659,6661,6673,6679,6689,6691,6701,6703,6709,6719,6733,6737,6761,6763,6779,6781,6791,6793,6803,6823,6827,6829,6833,6841,6857,6863,6869,6871,6883,6899,6907,6911,6917,6947,6949,6959,6961,6967,6971,6977,6983,6991,6997,7001,7013,7019,7027,7039,7043,7057,7069,7079,7103,7109,7121,7127,7129,7151,7159,7177,7187,7193,7207,7211,7213,7219,7229,7237,7243,7247,7253,7283,7297,7307,7309,7321,7331,7333,7349,7351,7369,7393,7411,7417,7433,7451,7457,7459,7477,7481,7487,7489,7499,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723,7727,7741,7753,7757,7759,7789,7793,7817,7823,7829,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919]

        # pi constant
        self.pi_num = math.pi
    
    def to_float(self, frac_str): # 분수 형태의 input string도 처리.
        try:
            return float(frac_str)
        except ValueError:
            num, denom = frac_str.split('/')
            try:
                leading, num = num.split(' ')
                whole = float(leading)
            except ValueError:
                whole = 0
            frac = float(num) / float(denom)
            return whole - frac if whole < 0 else whole + frac


    def is_number(self, value):
        try:
            self.to_float(value)
            return True
        except ValueError:
            return False

    def is_fraction(self, value):
        try:
            float(value)
            return False
        except:
            return True

    # convert function
    @timeout(10)
    def convert(self, postfix_eq):
        # 주요 function : 주어진 postfix 식을 code로 생성.
        self.__init__()
        operand_operator_list = postfix_eq.split()
        for n, i in enumerate(operand_operator_list):
            if i in self.operator_dic: # if operator
                operator_name = i
                operator_info = self.operator_dic[i]
                
                # list exceptional functions - init
                if operator_info[2] == 'list':
                    if i == '[OP_LIST_SOL]':
                        self.operand_stack.push('SOL', 'SOL')
                    elif i == '[OP_LIST_EOL]':
                        new_list = []
                        list_name = self.list_names.pop(0)
                        self.code_string += '{}= []\n'.format(list_name)
                        while True:
                            element, var_name = self.operand_stack.pop()
                            if element == 'SOL':
                                new_list.reverse()
                                self.code_string += '{}.reverse()\n'.format(list_name)
                                self.list_stack.push(new_list, list_name) # list의 stack은 따로 유지.
                                break
                            if '/' in str(element):
                                element = eval(str(element))
                            new_list.append(element)
                            self.code_string += 'if "/" in str({var_name}):\n    {var_name} = eval(str({var_name}))\n{list_name}.append({var_name})\n'.format(list_name=list_name, var_name=var_name)
                    
                    elif i == '[OP_LIST_POP]':
                        self.list_stack.pop()
                    else:
                        print("not defined")
                    
                elif operator_info[2] == 'aux':
                    if i == '[OP_MEM]':
                        arg2, _ = self.operand_stack.pop()
                        arg1, arg1_name = self.operand_stack.pop()
                        self.operand_stack.push(arg1, arg1_name)
                        if arg2.isidentifier():  # python 변수명으로 사용할 수 없으면 무시.
                            self.mem[arg2] = (arg1_name, arg1)
                            self.code_string += f"{arg2} = {arg1_name}\n"
                            
                # 사칙연산을 포함한 infix형태의 연산들 : + - * / // % **
                elif operator_info[2] == 'infix':
                    b, b_name = self.operand_stack.pop()
                    a, a_name = self.operand_stack.pop()

                    var_name = self.operand_names.pop(0)
                    self.code_string += '{var} = {a} {operator} {b}\n'.format(var = var_name, a=a_name, operator=operator_info[0], b=b_name)
                    intermediate_eq = "{}".format(str(a) + operator_info[0] +str(b))
                    intermediate = eval(intermediate_eq)
                    self.operand_stack.push(intermediate, var_name)

                # function: math.perm(num1, num2) etc.
                elif operator_info[2] == 'function':
                    var_name = self.operand_names.pop(0)
                    if operator_info[1]==0:
                        if operator_name == '[OP_GET_PI]':
                            intermediate = math.pi
                            self.code_string += '{var}=math.pi\n'.format(var=var_name)
                        else: #
                            print("not defined")
                    elif operator_info[1]==1: # OP_REVERSE_NUM
                        if operator_name == '[OP_REVERSE_NUM]':
                            a, a_name = self.operand_stack.pop()
                            intermediate_eq = str(operator_info[0]+'('+str(a)+')')
                            intermediate = eval(intermediate_eq)
                            self.code_string += '{} = int(str({})[::-1])\n'.format(var_name, a_name)
                        else:
                            a, a_name = self.operand_stack.pop()
                            intermediate_eq = str(operator_info[0]+'('+str(a)+')')
                            intermediate = eval(intermediate_eq)
                            self.code_string += '{} = {}({})\n'.format(var_name, operator_info[0], a_name)

                    elif operator_info[1]==2: # OP_COMB, OP_PERM
                        if operator_name == '[OP_GET_NTH_DECIMAL]':
                            b, b_name = self.operand_stack.pop()
                            b = int(b)
                            a, a_name = self.operand_stack.pop()
                            a = self.to_float(a)
                            intermediate = int((a * 10**b) % 10)
                            self.code_string += '{var} = int(({a} * 10**{b}) % 10)\n'.format(var=var_name, a=a_name, b=b_name)
                        elif operator_name == '[OP_GCD]':
                            num1, num1_name = self.operand_stack.pop()
                            num2, num2_name = self.operand_stack.pop()
                            num1, num2 = int(num1), int(num2)

                            intermediate = math.gcd(num1, num2) 
                            self.code_string += '{new_var} = math.gcd(int({var1}), int({var2}))\n'.format(new_var=var_name, var1=num1_name, var2=num2_name)
                        elif operator_name == '[OP_LCM]':
                            num1, num1_name = self.operand_stack.pop()
                            num2, num2_name = self.operand_stack.pop()
                            num1, num2 = int(num1), int(num2)

                            intermediate = num1 * num2 / math.gcd(num1, num2)
                            self.code_string += '{new_var} = {var1} * {var2} / math.gcd(int({var1}), int({var2}))\n'.format(new_var=var_name, var1=num1_name, var2=num2_name)

                        elif operator_name in ['[OP_CEIL]', '[OP_FLOOR]']:
                            b, b_name = self.operand_stack.pop()
                            b = int(b)
                            a, a_name = self.operand_stack.pop()
                            if operator_name == '[OP_CEIL]':
                                try:
                                    # 정수 올림
                                    int(a)
                                    intermediate_eq = 'int((({a}+9*10**({b}-2))//(10**({b}-1)))*10**({b}-1))\n'.format(a=a_name, b=b_name)
                                    intermediate = eval('int((({a}+9*10**({b}-2))//(10**({b}-1)))*10**({b}-1))\n'.format(a=a, b=b))
                                    self.code_string += '{var}={eq}\n'.format(var=var_name, eq=intermediate_eq)
                                except:
                                    # int(float) -> floor / int(float+1) -> ceil
                                    intermediate_eq = 'int({a}*10**{b}+1)/10**{b}\n'.format(a=a_name, b=b_name)
                                    intermediate = eval('int({a}*10**{b}+1)/10**{b}\n'.format(a=a, b=b))
                                    self.code_string += '{var}={eq}\n'.format(var=var_name, eq=intermediate_eq)
                            else: # [OP_FLOOR]
                                try:
                                    int(a)
                                    intermediate_eq = 'int(({a}//(10**({b}-1)))*10**({b}-1))\n'.format(a=a_name, b=b_name)
                                    intermediate = eval('int(({a}//(10**({b}-1)))*10**({b}-1))\n'.format(a=a, b=b))
                                    self.code_string += '{var}={eq}\n'.format(var=var_name, eq=intermediate_eq)
                                except:
                                    intermediate_eq = 'int({a}*10**{b})/10**{b}\n'.format(a=a_name, b=b_name)
                                    intermediate = eval('int({a}*10**{b})/10**{b}\n'.format(a=a, b=b))
                                    self.code_string += '{var}={eq}\n'.format(var=var_name, eq=intermediate_eq)
                        elif operator_name == '[OP_ROUND]':
                            b, b_name = self.operand_stack.pop()
                            b = int(b)
                            a, a_name = self.operand_stack.pop()
                            try:
                                int(str(a))
                                round_tgt = int(a//10**(b-2))%10
                                if round_tgt >= 5:
                                    intermediate = int(((a+9*10**(b-2))//(10**(b-1)))*10**(b-1))
                                else:
                                    intermediate = int((a//(10**(b-1)))*10**(b-1))
                                self.code_string += "round_tgt = int({a}//10**({b}-2)%10)\n\
if round_tgt >= 5:\n\
    {intermediate} = int((({a}+9*10**({b}-2))//(10**({b}-1)))*10**({b}-1))\n\
else:\n\
    {intermediate} = int(({a}//(10**({b}-1)))*10**({b}-1))".format(a=a, b=b, intermediate=var_name)
                            except:
                                a = self.to_float(a)
                                intermediate = round(a+1e-10, b) # epsilon 추가해야 정확한 값이 나옴. 1.7325 -> round(1.7325, 3) -> 1.732 / round(1.7325000001, 3) -> 1.733
                                self.code_string += '{var} = round(float({a})+1e-10, {b})\n'.format(var=var_name, a=a_name, b=b_name)
                        elif operator_name == '[OP_DAY_DIFF]':
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            intermediate = ['월', '화', '수', '목', '금', '토', '일']
                            intermediate = intermediate.index(b) - intermediate.index(a)
                            if intermediate >= 0:
                                intermediate =  intermediate + 1
                            else:
                                intermediate = (7 + intermediate) + 1
                            self.code_string += "{intermediate} = ['월', '화', '수', '목', '금', '토', '일']\n\
{intermediate} = {intermediate}.index({b}) - {intermediate}.index({a})\n\
if {intermediate} >= 0:\n\
    {intermediate} =  {intermediate} + 1\n\
else:\n\
        {intermediate} = (7 + {intermediate}) + 1\n".format(intermediate=var_name, a=a_name, b=b_name)
                        elif operator_name == '[OP_DUP_COMB]':
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            # try:
                            #     intermediate = math.comb(num1+num2-1, num2)
                            #     self.code_string += '{var} = math.comb(int({var1})+int({var2})-1, int({var2}))'.format(var=var_name, var1=num1_name, var2=num2_name)
                            # except:
                            #     if num1 > 10 or num2 > 10:
                            #         print("Memory issue")
                            #         return -1, self.code_string
                            #     intermediate = len(list(itertools.combinations([i for i in range(num1+num2-1)], num2)))
                            #     self.code_string += '{var} = len(list(itertools.combinations([i for i in range(int({var1})+int({var2})-1)], int({var2}))))\n'.format(var=var_name, var1=num1_name, var2=num2_name)
                            intermediate =1
                            a = int(a)
                            b = int(b)
                            a = a + b - 1
                            for i, elem in enumerate(range(b)):
                                intermediate = intermediate * (a-i)
                            for i, elem in enumerate(range(b)):
                                intermediate = intermediate / (i+1)
                            self.code_string += '{new_var} = 1\n\
{a} = int({a})\n\
{b} = int({b})\n\
{a} = {a} + {b} - 1\n\
for i, elem in enumerate(range({b})):\n\
    {new_var} = {new_var} * ({a}-i)\n\
for i, elem in enumerate(range({b})):\n\
    {new_var} = {new_var} / (i+1)\n'.format(new_var=var_name, a=a_name, b=b_name)
                        elif operator_name == '[OP_COMB]':
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            # try:
                            #     intermediate = math.comb(int(a),int(b))
                            #     self.code_string += '{var} = math.comb(int({a}),int({b}))'.format(var=var_name, a=a_name, b=b_name)
                            # except:
                            #     if int(a) > 10 or int(b) > 10:
                            #         print("Memory issue")
                            #         return -1, self.code_string
                            #     intermediate = len(list(itertools.combinations([i for i in range(int(a))], int(b))))
                            #     self.code_string += '{var} = len(list(itertools.combinations([i for i in range(int({a}))], int({b}))))\n'.format(var=var_name, a=a_name, b=b_name)
                            intermediate = 1
                            a = int(a)
                            b = int(b)
                            for i, elem in enumerate(range(b)):
                                intermediate = intermediate * (a-i)
                            for i, elem in enumerate(range(b)):
                                intermediate = intermediate / (i+1)
                            self.code_string += '{new_var} = 1\n\
{a} = int({a})\n\
{b} = int({b})\n\
for i, elem in enumerate(range({b})):\n\
    {new_var} = {new_var} * ({a}-i)\n\
for i, elem in enumerate(range({b})):\n\
    {new_var} = {new_var} / (i+1)\n'.format(new_var=var_name, a=a_name, b=b_name)
                        elif operator_name == '[OP_PERM]':
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            # try:
                            #     intermediate = math.perm(int(a),int(b))
                            #     self.code_string += '{var} = math.perm(int({a}),int({b}))'.format(var=var_name, a=a_name, b=b_name)
                            # except:
                            #     if int(a) > 10 or int(b) > 10:
                            #         print("Memory issue")
                            #         return -1, self.code_string
                            #     intermediate = len(list(itertools.permutations([i for i in range(int(a))], int(b))))
                            #     self.code_string += '{var} = len(list(itertools.permutations([i for i in range(int({a}))], int({b}))))\n'.format(var=var_name, a=a_name, b=b_name)
                            intermediate = 1
                            a = int(a)
                            b = int(b)
                            for i, elem in enumerate(range(b)):
                                intermediate = intermediate * (a-i)
                            self.code_string += '{new_var} = 1\n\
{a} = int({a})\n\
{b} = int({b})\n\
for i, elem in enumerate(range({b})):\n\
    {new_var} = {new_var} * ({a}-i)\n'.format(new_var=var_name, a=a_name, b=b_name)
                        else: # 현재 사용되지는 않는 부분. 기존 OP_COMB, OP_PERM 사용되던 부분. Python 3.8이상에서만. 다른 eq 추가 가능.
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            intermediate_eq = str(operator_info[0])+'('+a_name+','+b_name+')'
                            intermediate = eval(intermediate_eq)
                            self.code_string += '{var} = {intermediate}\n'.format(var=var_name, intermediate=intermediate_eq)
                    else:
                        print("not defined")
                    self.operand_stack.push(intermediate, var_name)
                elif operator_info[2] == 'function_special':
                    if operator_name == '[OP_DIGIT_UNK_SOLVER]':
                        x, x_name = self.operand_stack.pop()
                        eq, eq_name = self.operand_stack.pop()
                        eq = str(eq)
                        eq = eq.replace('×','*')
                        eq = eq.replace('x','*')
                        eq = eq.replace('÷','/')
                        ans_dict = dict()
                        variable_candi = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
                        ans_dict = {v:[] for v in set(eq) & variable_candi}
                        candi = list(itertools.product('0123456789', repeat=len(ans_dict)))
                        for c in candi:
                            temp = eq
                            for i, (k, _) in enumerate(ans_dict.items()):
                                temp = temp.replace(k, str(c[i]))
                            term_list = []
                            op_list = []
                            temp_c = ''
                            for tc in temp:
                                if tc not in '+-*/=><().':
                                    temp_c += tc
                                else:
                                    op_list.append(tc)
                                    term_list.append(temp_c)
                                    temp_c = ''
                            term_list.append(temp_c)
                            new_eq = ''
                            for i in range(len(op_list)):
                                new_eq += str(int(term_list[i]))+op_list[i]
                            new_eq += str(int(term_list[-1]))
                            if len(new_eq) == len(eq):
                                new_eq=new_eq.replace('=', '==')
                                new_eq=new_eq.replace('>==', '>=')
                                new_eq=new_eq.replace('<==', '<=')
                                eval_result = False
                                try:
                                    eval_result = eval(new_eq)
                                except:
                                    pass
                                if eval_result:
                                    for i, (k, _) in enumerate(ans_dict.items()):
                                        ans_dict[k].append(int(c[i]))

                        intermediate = list(set(ans_dict[x]))
                        if len(intermediate) == 1:
                            intermediate = intermediate[0]

                        if isinstance(intermediate, list):
                            new_list_name = self.list_names.pop(0)
                            self.list_stack.push(intermediate, new_list_name)
                            self.code_string += "ans_dict = dict()\n\
{eq} = {eq}.replace('×','*')\n\
{eq} = {eq}.replace('x','*')\n\
{eq} = {eq}.replace('÷','/')\n\
variable_candi = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])\n\
for v in set({eq}):\n\
    if v in variable_candi:\n\
        ans_dict[v] = []\n\
candi = list(itertools.product('0123456789', repeat=len(ans_dict)))\n\
for c in candi:\n\
    temp = {eq}\n\
    for i, (k, _) in enumerate(ans_dict.items()):\n\
        temp = temp.replace(k, str(c[i]))\n\
    term_list = []\n\
    op_list = []\n\
    temp_c = ''\n\
    for tc in temp:\n\
        if tc not in '+-*/=><().':\n\
            temp_c += tc\n\
        else:\n\
            op_list.append(tc)\n\
            term_list.append(temp_c)\n\
            temp_c = ''\n\
    term_list.append(temp_c)\n\
    new_eq = ''\n\
    for i in range(len(op_list)):\n\
        new_eq += str(int(term_list[i]))+op_list[i]\n\
    new_eq += str(int(term_list[-1]))\n\
    if len(new_eq) == len({eq}):\n\
        new_eq=new_eq.replace('=', '==')\n\
        new_eq=new_eq.replace('>==', '>=')\n\
        new_eq=new_eq.replace('<==', '<=')\n\
        eval_result = False\n\
        try:\n\
            eval_result = eval(new_eq)\n\
        except:\n\
            pass\n\
        if eval_result:\n\
            for i, (k, _) in enumerate(ans_dict.items()):\n\
                ans_dict[k].append(int(c[i]))\n\
{intermediate} = list(set(ans_dict[{x}]))\n".format(intermediate=new_list_name, eq=eq_name, x=x_name)
                        else:
                            new_var_name = self.operand_names.pop(0)
                            self.operand_stack.push(intermediate, new_var_name)
                            self.code_string += "ans_dict = dict()\n\
{eq} = {eq}.replace('×','*')\n\
{eq} = {eq}.replace('x','*')\n\
{eq} = {eq}.replace('÷','/')\n\
variable_candi = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])\n\
for v in set({eq}):\n\
    if v in variable_candi:\n\
        ans_dict[v] = 1\n\
candi = list(itertools.product('0123456789', repeat=len(ans_dict)))\n\
for c in candi:\n\
    temp = {eq}\n\
    for i, (k, _) in enumerate(ans_dict.items()):\n\
        temp = temp.replace(k, str(c[i]))\n\
    term_list = []\n\
    op_list = []\n\
    temp_c = ''\n\
    for tc in temp:\n\
        if tc not in '+-*/=><().':\n\
            temp_c += tc\n\
        else:\n\
            op_list.append(tc)\n\
            term_list.append(temp_c)\n\
            temp_c = ''\n\
    term_list.append(temp_c)\n\
    new_eq = ''\n\
    for i in range(len(op_list)):\n\
        new_eq += str(int(term_list[i]))+op_list[i]\n\
    new_eq += str(int(term_list[-1]))\n\
    if len(new_eq) == len({eq}):\n\
        new_eq=new_eq.replace('=', '==')\n\
        new_eq=new_eq.replace('>==', '>=')\n\
        new_eq=new_eq.replace('<==', '<=')\n\
        eval_result = False\n\
        try:\n\
            eval_result = eval(new_eq)\n\
        except:\n\
            pass\n\
        if eval_result:\n\
            for i, (k, _) in enumerate(ans_dict.items()):\n\
                ans_dict[k] = int(c[i])\n\
{intermediate} = ans_dict[{x}]\n".format(intermediate=new_var_name, eq=eq_name, x=x_name)
                    elif operator_name == '[OP_NUM_UNK_SOLVER]':
                        x, x_name = self.operand_stack.pop()
                        eq, eq_name = self.operand_stack.pop()
                        eq = str(eq)
                        eq = eq.replace('×','*')
                        eq = eq.replace('x','*')
                        eq = eq.replace('÷','/')
                        ans_dict = dict()
                        variable_candi = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
                        ans_dict = {v:[] for v in set(eq) & variable_candi}
                        candidate_num = [i for i in range(51)]
                        candi = list(itertools.product(candidate_num, repeat=len(ans_dict)))
                        for c in candi:
                            temp = eq
                            for i, (k, _) in enumerate(ans_dict.items()):
                                temp = temp.replace(k, str(c[i]))
                            term_list = []
                            op_list = []
                            temp_c = ''
                            for tc in temp:
                                if tc not in '+-*/=><().':
                                    temp_c += tc
                                else:
                                    op_list.append(tc)
                                    term_list.append(temp_c)
                                    temp_c = ''
                            term_list.append(temp_c)
                            new_eq = ''
                            for i in range(len(op_list)):
                                if term_list[i] == '':
                                    new_eq += str(term_list[i])+op_list[i]
                                else:
                                    new_eq += str(int(term_list[i]))+op_list[i]
                            new_eq += str(int(term_list[-1]))
                            new_eq=new_eq.replace('=', '==')
                            new_eq=new_eq.replace('>==', '>=')
                            new_eq=new_eq.replace('<==', '<=')
                            eval_result = False
                            try:
                                eval_result = eval(new_eq)
                            except:
                                pass
                            if eval_result:
                                for i, (k, _) in enumerate(ans_dict.items()):
                                    ans_dict[k].append(int(c[i]))

                        intermediate = list(set(ans_dict[x]))
                        if len(intermediate) == 1:
                            intermediate = intermediate[0]

                        if isinstance(intermediate, list):
                            new_list_name = self.list_names.pop(0)
                            self.list_stack.push(intermediate, new_list_name)
                            self.code_string += "ans_dict = dict()\n\
{eq} = {eq}.replace('×','*')\n\
{eq} = {eq}.replace('x','*')\n\
{eq} = {eq}.replace('÷','/')\n\
variable_candi = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])\n\
for v in set({eq}):\n\
    if v in variable_candi:\n\
        ans_dict[v] = []\n\
candidate_num = [i for i in range(51)]\n\
candi = list(itertools.product(candidate_num, repeat=len(ans_dict)))\n\
for c in candi:\n\
    temp = {eq}\n\
    for i, (k, _) in enumerate(ans_dict.items()):\n\
        temp = temp.replace(k, str(c[i]))\n\
    term_list = []\n\
    op_list = []\n\
    temp_c = ''\n\
    for tc in temp:\n\
        if tc not in '+-*/=><().':\n\
            temp_c += tc\n\
        else:\n\
            op_list.append(tc)\n\
            term_list.append(temp_c)\n\
            temp_c = ''\n\
    term_list.append(temp_c)\n\
    new_eq = ''\n\
    for i in range(len(op_list)):\n\
        if term_list[i] == '':\n\
            new_eq += str(term_list[i])+op_list[i]\n\
        else:\n\
            new_eq += str(int(term_list[i]))+op_list[i]\n\
    new_eq += str(int(term_list[-1]))\n\
    new_eq=new_eq.replace('=', '==')\n\
    new_eq=new_eq.replace('>==', '>=')\n\
    new_eq=new_eq.replace('<==', '<=')\n\
    eval_result = False\n\
    try:\n\
        eval_result = eval(new_eq)\n\
    except:\n\
        pass\n\
    if eval_result:\n\
        for i, (k, _) in enumerate(ans_dict.items()):\n\
            ans_dict[k].append(int(c[i]))\n\
{intermediate} = list(set(ans_dict[{x}]))\n".format(intermediate=new_list_name, eq=eq_name, x=x_name)
                        else:
                            new_var_name = self.operand_names.pop(0)
                            self.operand_stack.push(intermediate, new_var_name)
                            self.code_string += "ans_dict = dict()\n\
{eq} = {eq}.replace('×','*')\n\
{eq} = {eq}.replace('x','*')\n\
{eq} = {eq}.replace('÷','/')\n\
variable_candi = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])\n\
for v in set({eq}):\n\
    if v in variable_candi:\n\
        ans_dict[v] = 0\n\
candidate_num = [i for i in range(51)]\n\
candi = list(itertools.product(candidate_num, repeat=len(ans_dict)))\n\
for c in candi:\n\
    temp = {eq}\n\
    for i, (k, _) in enumerate(ans_dict.items()):\n\
        temp = temp.replace(k, str(c[i]))\n\
    term_list = []\n\
    op_list = []\n\
    temp_c = ''\n\
    for tc in temp:\n\
        if tc not in '+-*/=><().':\n\
            temp_c += tc\n\
        else:\n\
            op_list.append(tc)\n\
            term_list.append(temp_c)\n\
            temp_c = ''\n\
    term_list.append(temp_c)\n\
    new_eq = ''\n\
    for i in range(len(op_list)):\n\
        if term_list[i] == '':\n\
            new_eq += str(term_list[i])+op_list[i]\n\
        else:\n\
            new_eq += str(int(term_list[i]))+op_list[i]\n\
    new_eq += str(int(term_list[-1]))\n\
    new_eq=new_eq.replace('=', '==')\n\
    new_eq=new_eq.replace('>==', '>=')\n\
    new_eq=new_eq.replace('<==', '<=')\n\
    eval_result = False\n\
    try:\n\
        eval_result = eval(new_eq)\n\
    except:\n\
        pass\n\
    if eval_result:\n\
        for i, (k, _) in enumerate(ans_dict.items()):\n\
            ans_dict[k] = int(c[i])\n\
{intermediate} = ans_dict[{x}]\n".format(intermediate=new_var_name, eq=eq_name, x=x_name)
                    elif operator_name == '[OP_GEN_POSSIBLE_LIST]':
                        unk, unk_name = self.operand_stack.pop()
                        unk = str(unk)
                        ans_dict = dict()
                        variable_candi = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
                        ans_dict = {v:0 for v in set(unk) & variable_candi}
                        candi = list(itertools.product('0123456789', repeat=len(ans_dict)))
                        intermediate_list = []
                        for c in candi:
                            temp = unk
                            for i, (k, _) in enumerate(ans_dict.items()):
                                temp = temp.replace(k, str(c[i]))
                            if len(unk) == len(str(int(temp))):
                                new_elem = int(temp)
                                intermediate_list.append(new_elem)
                        
                        new_list_name = self.list_names.pop(0)
                        self.operand_stack.push(unk, unk_name)
                        self.list_stack.push(intermediate_list, new_list_name)
                        self.code_string += "ans_dict = dict()\n\
{unk} = str({unk})\n\
{intermediate_list} = []\n\
variable_candi = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])\n\
for v in set({unk}):\n\
    if v in variable_candi:\n\
        ans_dict[v] = 0\n\
candi = list(itertools.product('0123456789', repeat=len(ans_dict)))\n\
for c in candi:\n\
    temp = {unk}\n\
    for i, (k, _) in enumerate(ans_dict.items()):\n\
        temp = temp.replace(k, str(c[i]))\n\
    if len({unk}) == len(str(int(temp))):\n\
        new_elem = int(temp)\n\
        {intermediate_list}.append(new_elem)\n".format(unk=unk_name, intermediate_list=new_list_name)

                elif operator_info[2] == 'list_function':
                    if operator_info[1]==0: # OP_LIST_PRIME
                        if operator_name == '[OP_LIST_PRIME]':
                            new_list = self.prime_num
                            new_list_name = self.list_names.pop(0)
                            self.list_stack.push(new_list, new_list_name)
                            self.code_string += '{} = {}\n'.format(new_list_name, new_list)
                    elif operator_info[1]==1: # OP_LIST_SUM, OP_LIST_MEAN, OP_LIST_LEN, OP_LIST_GET_DIVISOR / input: list / output: scalar
                        if operator_name == '[OP_LIST_DISTINCT]':
                            temp_list, temp_lname = self.list_stack.pop()
                            intermediate_list =  list(set(temp_list))
                            new_list_name = self.list_names.pop(0)
                            self.code_string += '{} = list(set({}))\n'.format(new_list_name, temp_lname)
                            self.list_stack.push(temp_list, temp_lname)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_RANDOMGAME]':
                            temp_list, temp_lname = self.list_stack.pop()
                            intermediate_list = []
                            for i in range(len(temp_list)//2):
                                for j in range(int(temp_list[i*2+1])):
                                    intermediate_list.append(str(temp_list[i*2]))
                            new_list_name = self.list_names.pop(0)
                            self.code_string += "{intermediate_list} = []\n\
for i in range(len({temp_list})//2):\n\
    for j in range(int({temp_list}[i*2+1])):\n\
        {intermediate_list}.append(str({temp_list}[i*2]))\n".format(intermediate_list=new_list_name, temp_list=temp_lname)
                            self.list_stack.push(temp_list, temp_lname)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_MEAN]':
                            temp_list, temp_lname = self.list_stack.pop()
                            temp_list = [self.to_float(i) for i in temp_list]
                            intermediate = sum(temp_list) / len(temp_list)
                            new_var_name = self.operand_names.pop(0)
                            self.code_string += '{temp_list} = [float(i) for i in {temp_list}]\n\
{new_var_name} = sum({temp_list})/len({temp_list})\n'.format(new_var_name=new_var_name, temp_list=temp_lname)
                            self.operand_stack.push(intermediate, new_var_name)
                            self.list_stack.push(temp_list, temp_lname)
                        elif operator_name == '[OP_LIST_SUM]':
                            temp_list, temp_lname = self.list_stack.pop()
                            temp_list = [self.to_float(i) for i in temp_list]
                            intermediate = sum(temp_list)
                            new_var_name = self.operand_names.pop(0)
                            self.code_string += '{temp_list} = [float(i) for i in {temp_list}]\n\
{new_var_name} = sum({temp_list})\n'.format(new_var_name=new_var_name, temp_list=temp_lname)
                            self.operand_stack.push(intermediate, new_var_name)
                            self.list_stack.push(temp_list, temp_lname)
                        elif operator_name == '[OP_LIST2NUM]':
                            temp_list, temp_lname = self.list_stack.pop()
                            new_var_name = self.operand_names.pop(0)
                            intermediate = ''
                            for i in temp_list:
                                i = str(i)
                                intermediate = intermediate + i
                            self.code_string += '{new_var_name}=""\n\
for i in {temp_list}:\n\
    i = str(i)\n\
    {new_var_name} = {new_var_name} + i\n'.format(new_var_name = new_var_name, temp_list = temp_lname)
                            self.operand_stack.push(intermediate, new_var_name)
                            self.list_stack.push(temp_list,temp_lname)
                        elif operator_name == '[OP_NUM2LIST]':
                            a, a_name = self.operand_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            a = int(a)
                            while a//10 > 0:
                                intermediate_list.append(a%10)
                                a = a//10
                            intermediate_list.append(a%10)
                            intermediate_list = intermediate_list[::-1]
                            self.code_string += '{new_list_name} = []\n\
{a} = int({a})\n\
while {a}//10 > 0:\n\
    {new_list_name}.append({a}%10)\n\
    {a} = {a}//10\n\
{new_list_name}.append({a}%10)\n\
{new_list_name} = {new_list_name}[::-1]\n'.format(a=a_name, new_list_name=new_list_name)
                            self.list_stack.push(intermediate_list, new_list_name)
                            # print(intermediate_list)
                        elif operator_name == '[OP_LIST_INV]':
                            temp_list, temp_lname = self.list_stack.pop()
                            new_list_name = self.list_names.pop()
                            intermediate_list = temp_list[::-1]
                            self.code_string += '{intermediate_list} = {temp_list}[::-1]\n'.format(intermediate_list=new_list_name, temp_list=temp_lname)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_NUM2SUM]':
                            temp_list, temp_lname = self.list_stack.pop()
                            a_name = self.operand_names.pop(0)
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            for i in temp_list:
                                element = 0
                                i = int(i)
                                while i//10 > 0:
                                    element = element + i%10
                                    i = i//10
                                element = element + i%10
                                intermediate_list.append(element)
                            self.code_string += "{intermediate_list}=[]\n\
for i in {temp_list}:\n\
    {a_name} = 0\n\
    i = int(i)\n\
    while i//10 > 0:\n\
        {a_name} = {a_name} + i%10\n\
        i = i//10\n\
    {a_name} = {a_name} + i%10\n\
    {intermediate_list}.append({a_name})\n".format(intermediate_list=new_list_name, temp_list=temp_lname, a_name=a_name)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_GET_DIVISOR]':
                            num, num_name = self.operand_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            num = int(num)
                            intermediate_list = []
                            num_sqrt = int(math.sqrt(num))
                            for i in range(1, num_sqrt+1):
                                if num % i == 0:
                                    intermediate_list.append(i)
                                    intermediate_list.append(int(num/i))
                            new_list = sorted(set(intermediate_list))
                            self.list_stack.push(new_list, new_list_name)
                            self.code_string += "{intermediate_list} = []\n\
num_sqrt = int(math.sqrt({num_name}))\n\
for i in range(1, num_sqrt+1):\n\
    if {num_name} % i == 0:\n\
        {intermediate_list}.append(i)\n\
        {intermediate_list}.append(int({num_name}/i))\n\
{intermediate_list} = sorted(set({intermediate_list}))\n".format(num_name=num_name, intermediate_list=new_list_name)
                        else:
                            # print(self.list_stack.content_stack)
                            temp_list, temp_lname = self.list_stack.pop()
                            intermediate_eq = operator_info[0]+'('+str(temp_list)+')'
                            # intermediate_eq = str(operator_info[0]+'('+str(a)+')')
                            intermediate = eval(intermediate_eq)
                            new_var_name = self.operand_names.pop(0)
                            self.code_string += '{} = {}({})\n'.format(new_var_name, operator_info[0], temp_lname)
                            self.operand_stack.push(intermediate, new_var_name)
                            self.list_stack.push(temp_list, temp_lname)
                    elif operator_info[1]==2:
                        if operator_name in ['[OP_LIST_MAX]', '[OP_LIST_MIN]', '[OP_LIST_GET]', '[OP_LIST_INDEX]', '[OP_LIST_MORE]', '[OP_LIST_LESS]', '[OP_LIST_MORE_EQUAL]', '[OP_LIST_LESS_EQUAL]', \
                                             '[OP_LIST_GET_PERM]', '[OP_LIST_GET_PRODUCT]', '[OP_LIST_FINDSEQ]', '[OP_LIST_LEN_MOD_GET]']:# list하나와 scalar값 하나로 이루어진 function연산. / input: list, scalar / output: scalar, list
                            temp_list, temp_lname = self.list_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            try:
                                a = self.to_float(a)
                                a = self.intifint(a)
                            except:
                                pass
                            # a = self.operand_stack.pop()
                            if operator_name == '[OP_LIST_GET]': # list하나와 scalar값 하나로 이루어진 연산 / input: list, scalar / output: scalar
                                # print('[OP_LIST_GET]', temp_list)
                                intermediate = temp_list[a-1]
                                new_var_name = self.operand_names.pop(0)
                                self.code_string += '{} = {}[{}-1]\n'.format(new_var_name, temp_lname, a_name)
                                self.list_stack.push(temp_list, temp_lname)
                                self.operand_stack.push(intermediate, new_var_name)
                            elif operator_name == '[OP_LIST_INDEX]': # list하나와 scalar값 하나로 이루어진 연산 / input: list, scalar / output: scalar
                                if isinstance(a, int):
                                    try:
                                        try:
                                            intermediate = temp_list.index(str(a))+1
                                        except:
                                            intermediate = temp_list.index(str(float(a)))+1
                                    except:
                                        try:
                                            intermediate = temp_list.index(int(a))+1
                                        except:
                                            intermediate = temp_list.index(float(a))+1
                                
                                elif isinstance(a, float):
                                    try:
                                        intermediate = temp_list.index(str(a))+1
                                    except:
                                        intermediate = temp_list.index(float(a))+1
                                
                                else:
                                    intermediate = temp_list.index(str(a))+1

                                new_var_name = self.operand_names.pop(0)
                                self.code_string += '{} = {}.index({})+1\n'.format(new_var_name, temp_lname, a_name)
                                self.list_stack.push(temp_list, temp_lname)
                                self.operand_stack.push(intermediate, new_var_name)
                            elif operator_name in ['[OP_LIST_MORE]', '[OP_LIST_LESS]', '[OP_LIST_MORE_EQUAL]', '[OP_LIST_LESS_EQUAL]']:
                                new_list_name = self.list_names.pop(0)
                                if operator_name == '[OP_LIST_MORE]':
                                    intermediate_list = [i for i in temp_list if self.intifint(self.to_float(i)) > a]
                                    self.code_string += '{new_list} = []\n\
for i in {temp}:\n\
    if i > {a}:\n\
        {new_list}.append(i)\n'.format(new_list=new_list_name, temp=temp_lname, a=a_name)
                                elif operator_name == '[OP_LIST_LESS]':
                                    intermediate_list = [i for i in temp_list if self.intifint(self.to_float(i)) < a]
                                    # self.code_string += '{} = [i for i in {} if i < {}]\n'.format(new_list_name, temp_lname, a_name)
                                    self.code_string += '{new_list} = []\n\
for i in {temp}:\n\
    if i < {a}:\n\
        {new_list}.append(i)\n'.format(new_list=new_list_name, temp=temp_lname, a=a_name)
                                elif operator_name == '[OP_LIST_MORE_EQUAL]':
                                    intermediate_list = [i for i in temp_list if self.intifint(self.to_float(i)) >= a]
                                    # self.code_string += '{} = [i for i in {} if i >= {}]\n'.format(new_list_name, temp_lname, a_name)
                                    self.code_string += '{new_list} = []\n\
for i in {temp}:\n\
    if i >= {a}:\n\
        {new_list}.append(i)\n'.format(new_list=new_list_name, temp=temp_lname, a=a_name)
                                elif operator_name == '[OP_LIST_LESS_EQUAL]':
                                    intermediate_list = [i for i in temp_list if self.intifint(self.to_float(i)) <= a]
                                    # self.code_string += '{} = [i for i in {} if i <= {}]\n'.format(new_list_name, temp_lname, a_name)
                                    self.code_string += '{new_list} = []\n\
for i in {temp}:\n\
    if i <= {a}:\n\
        {new_list}.append(i)\n'.format(new_list=new_list_name, temp=temp_lname, a=a_name)
                                self.list_stack.push(temp_list, temp_lname)
                                self.list_stack.push(intermediate_list, new_list_name)
                            elif operator_name == '[OP_LIST_MAX]':
                                zizigo = temp_list.copy()
                                # for i in range(len(zizigo)):
                                #     zizigo[i] = float(zizigo[i])
                                zizigo.sort()
                                intermediate = zizigo[-a]
                                new_var_name = self.operand_names.pop(0)
                                new_list_name = self.list_names.pop(0)
                                self.code_string += '{new_list}={temp_list}.copy()\n{new_list}.sort()\n{intermediate} = {new_list}[-{a}]\n'.format(new_list=new_list_name,temp_list=temp_lname,intermediate=new_var_name,a=a_name)
                                self.list_stack.push(temp_list, temp_lname)
                                self.operand_stack.push(intermediate, new_var_name)
                            elif operator_name == '[OP_LIST_MIN]':
                                zizigo = temp_list.copy()
                                # for i in range(len(zizigo)):
                                #     zizigo[i] = float(zizigo[i])
                                zizigo.sort()
                                intermediate = temp_list[a-1]
                                new_var_name = self.operand_names.pop(0)
                                new_list_name = self.list_names.pop(0)
                                self.code_string += '{new_list}={temp_list}.copy()\n{new_list}.sort()\n{intermediate} = {new_list}[{a}-1]\n'.format(new_list=new_list_name,temp_list=temp_lname,intermediate=new_var_name,a=a_name)
                                self.list_stack.push(temp_list, temp_lname)
                                self.operand_stack.push(intermediate, new_var_name)
                            elif operator_name == '[OP_LIST_GET_PERM]': # 
                                intermediate_list = [str(i) for i in temp_list]
                                if len(intermediate_list) > 10 or int(a) > 10:
                                    print("Memory issue")
                                    return -1, self.code_string
                                new_list_name = self.list_names.pop(0)
                                intermediate_list = list(itertools.permutations(intermediate_list, a))
                                intermediate_list = [''.join(num_list) for num_list in intermediate_list]
                                intermediate_list = [str_num for str_num in intermediate_list if str_num[0] != '0']
                                self.code_string += "{intermediate_list} = [str(i) for i in {temp_list}]\n\
{intermediate_list} = list(itertools.permutations({intermediate_list}, {a}))\n\
{intermediate_list} = [''.join(num_list) for num_list in {intermediate_list}]\n\
{intermediate_list} = [str_num for str_num in {intermediate_list} if str_num[0] != '0']\n".format(intermediate_list=new_list_name, temp_list=temp_lname, a=a_name)
                                if self.is_number(intermediate_list[0]):
                                  intermediate_list = [self.to_float(i) for i in intermediate_list]
                                  self.code_string += "{intermediate_list} = [float(i) for i in {intermediate_list}]\n".format(intermediate_list = new_list_name)
                                
                                self.list_stack.push(temp_list, temp_lname)
                                self.list_stack.push(intermediate_list, new_list_name)
                                # print(intermediate_list)
                            elif operator_name == '[OP_LIST_GET_PRODUCT]': # 
                                intermediate_list = [str(i) for i in temp_list]
                                if len(intermediate_list) > 10 or int(a) > 6:
                                    print("Memory issue")
                                    return -1, self.code_string
                                intermediate_list = list(itertools.product(intermediate_list, repeat=a))
                                intermediate_list = [''.join(num_list) for num_list in intermediate_list]
                                intermediate_list = [str_num for str_num in intermediate_list if str_num[0] != '0']
                                new_list_name = self.list_names.pop(0)
                                self.code_string += "{intermediate_list} = [str(i) for i in {temp_list}]\n\
{intermediate_list} = list(itertools.product({intermediate_list}, repeat={a}))\n\
{intermediate_list} = [''.join(num_list) for num_list in {intermediate_list}]\n\
{intermediate_list} = [str_num for str_num in {intermediate_list} if str_num[0] != '0']\n".format(intermediate_list=new_list_name, temp_list=temp_lname, a=a_name)
                                if self.is_number(intermediate_list[0]):
                                  intermediate_list = [self.to_float(i) for i in intermediate_list]
                                  self.code_string += "{intermediate_list} = [float(i) for i in {intermediate_list}]\n".format(intermediate_list = new_list_name)
                                self.list_stack.push(temp_list, temp_lname)
                                self.list_stack.push(intermediate_list, new_list_name)
                                
                            elif operator_name == '[OP_LIST_FINDSEQ]': # [2,4,6,8] / [2,4,8] / [1,2,'a',8] / ['a', 6, 18, 54] / [1,2,4] / [1,2,4,7] / [1,2,4,8]
                                flag_unknown = False
                                for idx, i in enumerate(temp_list): # 수열 내에 숫자가 아닌 string이 있으면 미지수로 판단.
                                    if not self.is_number(i):
                                        flag_unknown=True
                                        unknown_idx = idx
                                        possible_idx = [i for i in range(len(temp_list))] # list of possible_idx를 만듬. unknown_idx만 제외.
                                        possible_idx.pop(unknown_idx)
                                        # 미지수는 하나만 있는 것으로 간주.
                                    else:
                                        temp_list[idx] = int(i)
                                
                                new_var_name = self.operand_names.pop(0)
                                # 미지수가 끼어있는 경우 OP_LIST_FINDSEQ의 n은 쓰이지 않음.
                                if flag_unknown: # 수열에 미지수가 끼어 있는 경우. - 등차, 등비만 구분 가능.
                                    # 가정 : 미지수가 낀 경우 수열의 길이는 최소 4임. 미지수를 제외하고 3개의 수는 존재. 3개만으로 수열 종류 판단.
                                    # 등차
                                    if (temp_list[possible_idx[2]] - temp_list[possible_idx[1]]) / (possible_idx[2]-possible_idx[1]) == (temp_list[possible_idx[1]] - temp_list[possible_idx[0]]) / (possible_idx[1]-possible_idx[0]):
                                        difference = int((temp_list[possible_idx[2]] - temp_list[possible_idx[1]])/ (possible_idx[2]-possible_idx[1]))
                                        start = temp_list[0] if unknown_idx != 0 else temp_list[1]-difference
                                        end = start + difference * 100 # arbitrary large number
                                        intermediate_list = [i for i in range(start, end, difference)]
                                        intermediate = intermediate_list[unknown_idx]

                                    # 등비
                                    elif (temp_list[possible_idx[2]] / temp_list[possible_idx[1]]) ** (possible_idx[1]-possible_idx[0]) == (temp_list[possible_idx[1]] / temp_list[possible_idx[0]]) ** (possible_idx[2]-possible_idx[1]):
                                        ratio = int(temp_list[possible_idx[2]] / temp_list[possible_idx[1]]) if possible_idx[2]-possible_idx[1]==1 else int(temp_list[possible_idx[1]] / temp_list[possible_idx[0]])
                                        start = temp_list[0] if unknown_idx != 0 else temp_list[1]/ratio
                                        end = 1000 # arbitrary large number
                                        intermediate_list = [start]
                                        for i in range(100): # 등비수열
                                            start *= ratio
                                            intermediate_list.append(start)
                                        intermediate = intermediate_list[unknown_idx]
                                    else: # 등차, 등비 아닌 경우.
                                        intermediate = -1
                                    self.list_stack.push(temp_list, temp_lname)
                                    self.operand_stack.push(intermediate, new_var_name)

                                else: # 사이에 미지수가 없는 경우. - 등차, 등비, 소수, 피보나치, 계차-등차, 계차-등비 구분 가능.
                                    if len(temp_list) >= 4: # list의 길이가 4 이상인 경우, 4개의 element를 사용하여 수열 규칙 확인.
                                        if temp_list[1]-temp_list[0] == temp_list[2]-temp_list[1] and temp_list[3]-temp_list[2] == temp_list[2]-temp_list[1]: # 등차수열
                                            difference = temp_list[1]-temp_list[0] # common difference / 공차
                                            start = temp_list[0]
                                            end = start + difference * 100 # arbitrary large number
                                            intermediate_list = [i for i in range(start, end, difference)]
                                            intermediate = intermediate_list[a-1]

                                        elif temp_list[1]/temp_list[0] == temp_list[2]/temp_list[1] and temp_list[3]/temp_list[2] == temp_list[2]/temp_list[1]: # 등비수열 / 가정: list의 element들은 int.
                                            ratio = temp_list[1]//temp_list[0] # common ratio / 공비
                                            start = temp_list[0]
                                            end = 1000 # arbitrary large number
                                            intermediate_list = [start]
                                            for i in range(100): # 등비수열
                                                start *= ratio
                                                intermediate_list.append(start)
                                            intermediate = intermediate_list[a-1]
                                        
                                        # from here, less likely.
                                        elif temp_list[0:4] == [2,3,5,7]: # prime number
                                            intermediate_list = self.prime_num
                                            intermediate = intermediate_list[a-1]

                                        elif temp_list[0]+temp_list[1] == temp_list[2] and temp_list[1]+temp_list[2] == temp_list[3]: # naive fibonacci - more general case.
                                            intermediate_list = [temp_list[0], temp_list[1], temp_list[2]]
                                            for i in range(100):
                                                intermediate_list.append(intermediate_list[i+1]+intermediate_list[i+2])
                                            intermediate = intermediate_list[a-1]

                                        elif (temp_list[2]-temp_list[1])-(temp_list[1]-temp_list[0]) == (temp_list[3]-temp_list[2])-(temp_list[2]-temp_list[1]): # 계차수열 - 계차 : 등차수열
                                            # seq_of_diff = [(temp_list[1]-temp_list[0]), (temp_list[2]-temp_list[1]), (temp_list[3]-temp_list[2]), ...]
                                            int_difference = (temp_list[2]-temp_list[1])-(temp_list[1]-temp_list[0])
                                            int_start = temp_list[1]-temp_list[0]
                                            int_end = int_start + int_difference * 100
                                            seq_of_diff = [i for i in range(int_start, int_end, int_difference)]
                                            intermediate_list = []
                                            for i in range(100):
                                                intermediate_list.append(temp_list[0]+sum(seq_of_diff[0:i]))
                                            intermediate = intermediate_list[a-1]

                                        elif (temp_list[2]-temp_list[1])/(temp_list[1]-temp_list[0]) == (temp_list[3]-temp_list[2])/(temp_list[2]-temp_list[1]): # 계차수열 - 계차 : 등비수열
                                            int_ratio = (temp_list[2]-temp_list[1])//(temp_list[1]-temp_list[0]) # common ratio / 공비
                                            int_start = temp_list[1]-temp_list[0]
                                            seq_of_ratio = [int_start]
                                            intermediate_list = []
                                            for i in range(100): # 등비수열
                                                int_start *= int_ratio
                                                seq_of_ratio.append(int_start)
                                            for i in range(100):
                                                intermediate_list.append(temp_list[0]+sum(seq_of_ratio[0:i]))
                                            intermediate = intermediate_list[a-1]
                                        elif len(temp_list) >= 5: # 계차-계차-등차 / 계차-계차-등비
                                            temp_a = (temp_list[4]-temp_list[3])-(temp_list[3]-temp_list[2])
                                            temp_b = (temp_list[3]-temp_list[2])-(temp_list[2]-temp_list[1])
                                            temp_c = (temp_list[2]-temp_list[1])-(temp_list[1]-temp_list[0])
                                            if temp_a-temp_b == temp_b-temp_c: #계차-계차-등차
                                                int_int_start = temp_c
                                                int_int_difference = temp_b-temp_c
                                                int_int_end = int_int_start + int_int_difference * 100
                                                seq_seq_difference = [i for i in range(int_int_start, int_int_end, int_int_difference)]
                                                seq_of_seq = []
                                                int_start = temp_list[1]-temp_list[0]
                                                for i in range(100):
                                                    seq_of_seq.append(int_start+sum(seq_seq_difference[0:i]))
                                                start = temp_list[0]
                                                intermediate_list=[]
                                                for i in range(100):
                                                    intermediate_list.append(start+sum(seq_of_seq[0:i]))
                                            elif temp_a/temp_b == temp_b/temp_c: # 계차-계차-등비
                                                int_int_start = temp_c
                                                int_int_ratio = temp_b/temp_c
                                                seq_seq_ratio = [int_int_start]
                                                for i in range(100):
                                                    int_int_start*=int_int_ratio
                                                    seq_seq_ratio.append(int_int_start)
                                                seq_of_seq = []
                                                int_start = temp_list[1]-temp_list[0]
                                                for i in range(100):
                                                    seq_of_seq.append(int_start+sum(seq_seq_ratio[0:i]))
                                                start = temp_list[0]
                                                intermediate_list=[]
                                                for i in range(100):
                                                    intermediate_list.append(start+sum(seq_of_seq[0:i]))
                                            intermediate = intermediate_list[a-1]
                                        else:
                                            intermediate = -1
                                        self.list_stack.push(temp_list, temp_lname)
                                        self.operand_stack.push(intermediate, new_var_name)

                                    else: 
                                    # list의 길이가 4 미만인 경우. 3개의 수를 확인하여 수열 확인. incomplete.
                                    # [1,2,4,7,11] 같은 경우 계차-등차수열이지만, 앞의 3개 숫자[1,2,4]만 확인했을 때는 등비수열로 착각할 수 있음.
                                        if len(temp_list) < 3: # 3개보다 작은 경우 수열 문제를 풀지 못함.
                                            intermediate = -1
                                        elif temp_list[1]-temp_list[0] == temp_list[2]-temp_list[1]: # 등차수열
                                            difference = temp_list[1]-temp_list[0] # common difference / 공차
                                            start = temp_list[0]
                                            end = start + difference * 100 # arbitrary large number
                                            intermediate_list = [i for i in range(start, end, difference)]
                                            intermediate = intermediate_list[a-1]

                                        elif temp_list[1]/temp_list[0] == temp_list[2]/temp_list[1]: # 등비수열 / 가정: list의 element들은 int.
                                            ratio = temp_list[1]//temp_list[0] # common ratio / 공비
                                            start = temp_list[0]
                                            end = 1000 # arbitrary large number
                                            intermediate_list = [start]
                                            for i in range(100): # 등비수열
                                                start *= ratio
                                                intermediate_list.append(start)
                                            intermediate = intermediate_list[a-1]
                                        else:
                                            intermediate = -1
                                        self.list_stack.push(temp_list, temp_lname)
                                        self.operand_stack.push(intermediate, new_var_name)
                                self.code_string += "flag_unknown = False\n\
for idx, i in enumerate({temp_list}):\n\
    try:\n\
        float(i)\n\
        is_number = True\n\
    except ValueError:\n\
        is_number = False\n\
    if not is_number:\n\
        flag_unknown = True\n\
        unknown_idx = idx\n\
        possible_idx = [i for i in range(len({temp_list}))]\n\
        possible_idx.pop(unknown_idx)\n\
    else:\n\
        {temp_list}[idx] = int(i)\n\
if flag_unknown:\n\
    if ({temp_list}[possible_idx[2]] - {temp_list}[possible_idx[1]]) / (possible_idx[2]-possible_idx[1]) == ({temp_list}[possible_idx[1]] - {temp_list}[possible_idx[0]]) / (possible_idx[1]-possible_idx[0]):\n\
        difference = int(({temp_list}[possible_idx[2]] - {temp_list}[possible_idx[1]])/ (possible_idx[2]-possible_idx[1]))\n\
        start = {temp_list}[0] if unknown_idx != 0 else {temp_list}[1]-difference\n\
        end = start + difference * 100\n\
        intermediate_list = [i for i in range(start, end, difference)]\n\
        intermediate = intermediate_list[unknown_idx]\n\
    elif ({temp_list}[possible_idx[2]] / {temp_list}[possible_idx[1]]) ** (possible_idx[1]-possible_idx[0]) == ({temp_list}[possible_idx[1]] / {temp_list}[possible_idx[0]]) ** (possible_idx[2]-possible_idx[1]):\n\
        ratio = int({temp_list}[possible_idx[2]] / {temp_list}[possible_idx[1]]) if possible_idx[2]-possible_idx[1]==1 else int({temp_list}[possible_idx[1]] / {temp_list}[possible_idx[0]])\n\
        start = {temp_list}[0] if unknown_idx != 0 else {temp_list}[1]/ratio\n\
        end = 1000\n\
        intermediate_list = [start]\n\
        for i in range(100):\n\
            start *= ratio\n\
            intermediate_list.append(start)\n\
        intermediate = intermediate_list[unknown_idx]\n\
    else:\n\
        intermediate = -1\n\
else:\n\
    if len({temp_list}) >= 4:\n\
        if {temp_list}[1]-{temp_list}[0] == {temp_list}[2]-{temp_list}[1] and {temp_list}[3]-{temp_list}[2] == {temp_list}[2]-{temp_list}[1]:\n\
            difference = {temp_list}[1]-{temp_list}[0]\n\
            start = {temp_list}[0]\n\
            end = start + difference * 100\n\
            intermediate_list = [i for i in range(start, end, difference)]\n\
            intermediate = intermediate_list[{a}-1]\n\
        elif {temp_list}[1]/{temp_list}[0] == {temp_list}[2]/{temp_list}[1] and {temp_list}[3]/{temp_list}[2] == {temp_list}[2]/{temp_list}[1]:\n\
            ratio = {temp_list}[1]//{temp_list}[0]\n\
            start = {temp_list}[0]\n\
            end = 1000\n\
            intermediate_list = [start]\n\
            for i in range(100):\n\
                start *= ratio\n\
                intermediate_list.append(start)\n\
            intermediate = intermediate_list[{a}-1]\n\
        elif {temp_list}[0:4] == [2,3,5,7]:\n\
            intermediate_list = self.prime_num\n\
            intermediate = intermediate_list[{a}-1]\n\
        elif {temp_list}[0]+{temp_list}[1] == {temp_list}[2] and {temp_list}[1]+{temp_list}[2] == {temp_list}[3]:\n\
            intermediate_list = [{temp_list}[0], {temp_list}[1], {temp_list}[2]]\n\
            for i in range(100):\n\
                intermediate_list.append(intermediate_list[i+1]+intermediate_list[i+2])\n\
            intermediate = intermediate_list[{a}-1]\n\
        elif ({temp_list}[2]-{temp_list}[1])-({temp_list}[1]-{temp_list}[0]) == ({temp_list}[3]-{temp_list}[2])-({temp_list}[2]-{temp_list}[1]):\n\
            int_difference = ({temp_list}[2]-{temp_list}[1])-({temp_list}[1]-{temp_list}[0])\n\
            int_start = {temp_list}[1]-{temp_list}[0]\n\
            int_end = int_start + int_difference * 100\n\
            seq_of_diff = [i for i in range(int_start, int_end, int_difference)]\n\
            intermediate_list = []\n\
            for i in range(100):\n\
                intermediate_list.append({temp_list}[0]+sum(seq_of_diff[0:i]))\n\
            intermediate = intermediate_list[{a}-1]\n\
        elif ({temp_list}[2]-{temp_list}[1])/({temp_list}[1]-{temp_list}[0]) == ({temp_list}[3]-{temp_list}[2])/({temp_list}[2]-{temp_list}[1]):\n\
            int_ratio = ({temp_list}[2]-{temp_list}[1])//({temp_list}[1]-{temp_list}[0])\n\
            int_start = {temp_list}[1]-{temp_list}[0]\n\
            seq_of_ratio = [int_start]\n\
            intermediate_list = []\n\
            for i in range(100):\n\
                int_start *= int_ratio\n\
                seq_of_ratio.append(int_start)\n\
            for i in range(100):\n\
                intermediate_list.append({temp_list}[0]+sum(seq_of_ratio[0:i]))\n\
            intermediate = intermediate_list[{a}-1]\n\
        elif len({temp_list}) >= 5:\n\
            temp_a = ({temp_list}[4]-{temp_list}[3])-({temp_list}[3]-{temp_list}[2])\n\
            temp_b = ({temp_list}[3]-{temp_list}[2])-({temp_list}[2]-{temp_list}[1])\n\
            temp_c = ({temp_list}[2]-{temp_list}[1])-({temp_list}[1]-{temp_list}[0])\n\
            if temp_a-temp_b == temp_b-temp_c:\n\
                int_int_start = temp_c\n\
                int_int_difference = temp_b-temp_c\n\
                int_int_end = int_int_start + int_int_difference * 100\n\
                seq_seq_difference = [i for i in range(int_int_start, int_int_end, int_int_difference)]\n\
                seq_of_seq = []\n\
                int_start = {temp_list}[1]-{temp_list}[0]\n\
                for i in range(100):\n\
                    seq_of_seq.append(int_start+sum(seq_seq_difference[0:i]))\n\
                start = {temp_list}[0]\n\
                intermediate_list=[]\n\
                for i in range(100):\n\
                    intermediate_list.append(start+sum(seq_of_seq[0:i]))\n\
            elif temp_a/temp_b == temp_b/temp_c:\n\
                int_int_start = temp_c\n\
                int_int_ratio = temp_b/temp_c\n\
                seq_seq_ratio = [int_int_start]\n\
                for i in range(100):\n\
                    int_int_start*=int_int_ratio\n\
                    seq_seq_ratio.append(int_int_start)\n\
                seq_of_seq = []\n\
                int_start = {temp_list}[1]-{temp_list}[0]\n\
                for i in range(100):\n\
                    seq_of_seq.append(int_start+sum(seq_seq_ratio[0:i]))\n\
                start = {temp_list}[0]\n\
                intermediate_list=[]\n\
                for i in range(100):\n\
                    intermediate_list.append(start+sum(seq_of_seq[0:i]))\n\
            intermediate = intermediate_list[{a}-1]\n\
        else:\n\
            intermediate = -1\n\
    else:\n\
        if len({temp_list}) < 3:\n\
            intermediate = -1\n\
        elif {temp_list}[1]-{temp_list}[0] == {temp_list}[2]-{temp_list}[1]:\n\
            difference = {temp_list}[1]-{temp_list}[0]\n\
            start = {temp_list}[0]\n\
            end = start + difference * 100\n\
            intermediate_list = [i for i in range(start, end, difference)]\n\
            intermediate = intermediate_list[{a}-1]\n\
        elif {temp_list}[1]/{temp_list}[0] == {temp_list}[2]/{temp_list}[1]:\n\
            ratio = {temp_list}[1]//{temp_list}[0]\n\
            start = {temp_list}[0]\n\
            end = 1000\n\
            intermediate_list = [start]\n\
            for i in range(100):\n\
                start *= ratio\n\
                intermediate_list.append(start)\n\
            intermediate = intermediate_list[{a}-1]\n\
        else:\n\
            intermediate = -1\n\
            intermediate = intermediate_list[{a}-1]\n\
{res} = intermediate\n".format(temp_list=temp_lname, res=new_var_name, a=a_name)

                            elif operator_name == '[OP_LIST_LEN_MOD_GET]':
                                intermediate = len(temp_list)
                                intermediate = int(a%intermediate)
                                intermediate = temp_list[intermediate]
                                new_var_name = self.operand_names.pop(0)
                                new_list_name = self.list_names.pop(0)
                                self.code_string += "{var} = len({temp_list})\n{var} = int({a}%{var})\n{var} = {temp_list}[{var}]\n".format(var=new_var_name, temp_list=temp_lname, a=a_name)
                                self.list_stack.push(temp_list, new_list_name)
                                self.operand_stack.push(intermediate, new_var_name)
                            else:
                                pass
                            
                        elif operator_name in ['[OP_SET_UNION]', '[OP_SET_INTERSECT]', '[OP_SET_DIFFERENCE]', '[OP_LIST_COND_MAX_MIN]']:# list 2개로 이루어진 연산. / input: list, list / output: list
                            b_list, b_lname = self.list_stack.pop()
                            a_list, a_lname = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            if operator_name == '[OP_SET_UNION]':
                                intermediate_list = list(set(a_list) | set(b_list))
                                self.code_string += '{} = list(set({}) | set({}))\n'.format(new_list_name, a_lname, b_lname)
                            elif operator_name == '[OP_SET_INTERSECT]':
                                intermediate_list = list(set(a_list) & set(b_list))
                                self.code_string += '{} = list(set({}) & set({}))\n'.format(new_list_name, a_lname, b_lname)
                            elif operator_name == '[OP_SET_DIFFERENCE]':
                                intermediate_list = list(set(a_list) - set(b_list))
                                self.code_string += '{} = list(set({}) - set({}))\n'.format(new_list_name, a_lname, b_lname)
                            elif operator_name == '[OP_LIST_COND_MAX_MIN]':
                                from queue import Queue
                                condition_list = b_list
                                condition_name = b_lname

                                entity_list = a_list
                                entity_name = a_lname

                                input_dict = dict()
                                for cnt, i in enumerate(entity_list):
                                    input_dict[i] = cnt
                                input_reverse_dict = dict()
                                for cnt, i in enumerate(entity_list):
                                    input_reverse_dict[cnt] = i
                                adj_mat = []
                                for _ in range(len(entity_list)):
                                    temp_list = []
                                    for _ in range(len(entity_list)):
                                        temp_list.append(0)
                                    adj_mat.append(temp_list)
                                is_visited = []
                                for _ in range(len(entity_list)):
                                    temp_list = []
                                    for _ in range(len(entity_list)):
                                        temp_list.append(0)
                                    is_visited.append(temp_list)
                                que = Queue()
                                iterate_num = len(condition_list)//3
                                for i in range(iterate_num):
                                    operand_1 = condition_list[i*3]
                                    operand_2 = condition_list[i*3 + 1]
                                    operand_1_id = input_dict[operand_1]
                                    operand_2_id = input_dict[operand_2]
                                    operator = condition_list[i*3+2]
                                    if operator == '>':
                                        adj_mat[operand_1_id][operand_2_id] = 1
                                        is_visited[operand_1_id][operand_2_id] = 1
                                        que.put((operand_1_id, operand_2_id))

                                        adj_mat[operand_2_id][operand_1_id] = -1
                                        is_visited[operand_2_id][operand_1_id] = 1
                                        que.put((operand_2_id, operand_1_id))
                                    elif operator == '<':
                                        adj_mat[operand_1_id][operand_2_id] = -1
                                        is_visited[operand_1_id][operand_2_id] = 1
                                        que.put((operand_1_id, operand_2_id))

                                        adj_mat[operand_2_id][operand_1_id] = 1
                                        is_visited[operand_2_id][operand_1_id] = 1
                                        que.put((operand_2_id, operand_1_id))
                                while not que.empty():
                                    operand_1, operand_2 = que.get()
                                    if adj_mat[operand_1][operand_2] == 1:
                                        for i in range(0, len(entity_list)):
                                            if (adj_mat[operand_1][i] == -1) and (not is_visited[operand_2][i]):
                                                adj_mat[operand_2][i] = -1
                                                adj_mat[i][operand_2] = 1
                                                is_visited[operand_2][i] = 1
                                                is_visited[i][operand_2] = 1
                                                que.put((operand_2, i))
                                                que.put((i, operand_2))
                                        for i in range(0, len(entity_list)):
                                            if (adj_mat[operand_2][i] == 1) and (not is_visited[operand_1][i]):
                                                adj_mat[operand_1][i] = 1
                                                adj_mat[i][operand_1] = -1
                                                is_visited[operand_1][i] = 1
                                                is_visited[i][operand_1] = 1
                                                que.put((operand_1, i))
                                                que.put((i, operand_1))
                                    if adj_mat[operand_1][operand_2] == -1:
                                        for i in range(0, len(entity_list)):
                                            if (adj_mat[operand_1][i] == 1) and (not is_visited[i][operand_2]):
                                                adj_mat[i][operand_2] = -1
                                                adj_mat[operand_2][i] = 1
                                                is_visited[i][operand_2] = 1
                                                is_visited[operand_2][i] = 1
                                                que.put((i, operand_2))
                                                que.put((operand_2, i))
                                        for i in range(0, len(entity_list)):
                                            if (adj_mat[operand_2][i] == -1) and (not is_visited[operand_1][i]):
                                                adj_mat[operand_1][i] = -1
                                                adj_mat[i][operand_1] = 1
                                                is_visited[i][operand_1] = 1
                                                is_visited[operand_1][i] = 1
                                                que.put((operand_1, i))
                                                que.put((i, operand_1))
                                sum_list = [sum(i) for i in adj_mat]
                                largest = -900
                                largest_index = -1
                                smallest = 900
                                smallest_index = -1
                                for cnt, i in enumerate(sum_list):
                                    if largest < i:
                                        largest = i
                                        largest_index = cnt
                                    if smallest > i:
                                        smallest = i
                                        smallest_index = cnt
                                intermediate_list = []
                                intermediate_list.append(input_reverse_dict[largest_index])
                                intermediate_list.append(input_reverse_dict[smallest_index])
                                self.code_string += "from queue import Queue\n\
input_dict = dict()\n\
for cnt, i in enumerate({entity_list}):\n\
    input_dict[i] = cnt\n\
input_reverse_dict = dict()\n\
for cnt, i in enumerate({entity_list}):\n\
    input_reverse_dict[cnt] = i\n\
adj_mat = []\n\
for _ in range(len({entity_list})):\n\
    temp_list = []\n\
    for _ in range(len({entity_list})):\n\
        temp_list.append(0)\n\
    adj_mat.append(temp_list)\n\
is_visited = []\n\
for _ in range(len({entity_list})):\n\
    temp_list = []\n\
    for _ in range(len({entity_list})):\n\
        temp_list.append(0)\n\
    is_visited.append(temp_list)\n\
que = Queue()\n\
iterate_num = len({condition_list})//3\n\
for i in range(iterate_num):\n\
    operand_1 = {condition_list}[i*3]\n\
    operand_2 = {condition_list}[i*3 + 1]\n\
    operand_1_id = input_dict[operand_1]\n\
    operand_2_id = input_dict[operand_2]\n\
    operator = {condition_list}[i*3+2]\n\
    if operator == '>':\n\
        adj_mat[operand_1_id][operand_2_id] = 1\n\
        is_visited[operand_1_id][operand_2_id] = 1\n\
        que.put((operand_1_id, operand_2_id))\n\
        adj_mat[operand_2_id][operand_1_id] = -1\n\
        is_visited[operand_2_id][operand_1_id] = 1\n\
        que.put((operand_2_id, operand_1_id))\n\
    elif operator == '<':\n\
        adj_mat[operand_1_id][operand_2_id] = -1\n\
        is_visited[operand_1_id][operand_2_id] = 1\n\
        que.put((operand_1_id, operand_2_id))\n\
        adj_mat[operand_2_id][operand_1_id] = 1\n\
        is_visited[operand_2_id][operand_1_id] = 1\n\
        que.put((operand_2_id, operand_1_id))\n\
while not que.empty():\n\
    operand_1, operand_2 = que.get()\n\
    if adj_mat[operand_1][operand_2] == 1:\n\
        for i in range(0, len({entity_list})):\n\
            if (adj_mat[operand_1][i] == -1) and (not is_visited[operand_2][i]):\n\
                adj_mat[operand_2][i] = -1\n\
                adj_mat[i][operand_2] = 1\n\
                is_visited[operand_2][i] = 1\n\
                is_visited[i][operand_2] = 1\n\
                que.put((operand_2, i))\n\
                que.put((i, operand_2))\n\
        for i in range(0, len({entity_list})):\n\
            if (adj_mat[operand_2][i] == 1) and (not is_visited[operand_1][i]):\n\
                adj_mat[operand_1][i] = 1\n\
                adj_mat[i][operand_1] = -1\n\
                is_visited[operand_1][i] = 1\n\
                is_visited[i][operand_1] = 1\n\
                que.put((operand_1, i))\n\
                que.put((i, operand_1))\n\
    if adj_mat[operand_1][operand_2] == -1:\n\
        for i in range(0, len({entity_list})):\n\
            if (adj_mat[operand_1][i] == 1) and (not is_visited[i][operand_2]):\n\
                adj_mat[i][operand_2] = -1\n\
                adj_mat[operand_2][i] = 1\n\
                is_visited[i][operand_2] = 1\n\
                is_visited[operand_2][i] = 1\n\
                que.put((i, operand_2))\n\
                que.put((operand_2, i))\n\
        for i in range(0, len({entity_list})):\n\
            if (adj_mat[operand_2][i] == -1) and (not is_visited[operand_1][i]):\n\
                adj_mat[operand_1][i] = -1\n\
                adj_mat[i][operand_1] = 1\n\
                is_visited[i][operand_1] = 1\n\
                is_visited[operand_1][i] = 1\n\
                que.put((operand_1, i))\n\
                que.put((i, operand_1))\n\
sum_list = [sum(i) for i in adj_mat]\n\
largest = -900\n\
largest_index = -1\n\
smallest = 900\n\
smallest_index = -1\n\
for cnt, i in enumerate(sum_list):\n\
    if largest < i:\n\
        largest = i\n\
        largest_index = cnt\n\
    if smallest > i:\n\
        smallest = i\n\
        smallest_index = cnt\n\
{intermediate_list} = []\n\
{intermediate_list}.append(input_reverse_dict[largest_index])\n\
{intermediate_list}.append(input_reverse_dict[smallest_index])\n".format(entity_list=entity_name, condition_list=condition_name, intermediate_list=new_list_name)
                            self.list_stack.push(intermediate_list, new_list_name) # SET 연산의 경우 set 연산 후 사용한 list는 삭제되고 결과값만 list stack에 저장됨.

                        elif operator_name in ['[OP_LIST_SCALAR_ADD]', '[OP_LIST_SCALAR_MUL]']:
                            a, a_name = self.operand_stack.pop()
                            temp_list, temp_lname = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            if operator_name == '[OP_LIST_SCALAR_ADD]':
                                intermediate_list = [i + int(a) for i in temp_list]
                                self.code_string += "{} = [i + {} for i in {}]\n".format(new_list_name, a_name, temp_lname)
                            else: # OP_LIST_SCALAR_MUL
                                intermediate_list = [i * int(a) for i in temp_list]
                                self.code_string += "{} = [i * {} for i in {}]\n".format(new_list_name, a_name, temp_lname)
                            self.list_stack.push(temp_list, temp_lname)
                            self.list_stack.push(intermediate_list, new_list_name)

                        elif operator_name == '[OP_LIST_DIVISIBLE]': # temp_list 에서 a로 나누어떨어지는 수들만 새로 list를 선언하여 return
                            a, a_name = self.operand_stack.pop()
                            temp_list, temp_lname = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            a = int(a)
                            for i in temp_list:
                                i =int(i)
                                if i % a == 0:
                                    intermediate_list.append(i)
                            self.code_string += "{intermediate_list} = []\n\
{a} = int({a})\n\
for i in {temp_list}:\n\
    i = int(i)\n\
    if i % {a} == 0:\n\
        {intermediate_list}.append(i)\n".format(intermediate_list=new_list_name, a=a_name, temp_list=temp_lname)
                            self.list_stack.push(temp_list, temp_lname)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_FIND_NUM]':
                            a, a_name = self.operand_stack.pop()
                            temp_list, temp_lname = self.list_stack.pop()
                            new_var_name = self.operand_names.pop(0)
                            intermediate = 0
                            a = int(a)
                            for i in temp_list:
                                i = int(i)
                                if i == a:
                                    intermediate = intermediate + 1
                            self.code_string += '{intermediate} = 0\n\
{a} = int({a})\n\
for i in {temp_list}:\n\
    i = int(i)\n\
    if i == {a}:\n\
        {intermediate} = {intermediate} + 1\n'.format(intermediate=new_var_name, temp_list=temp_lname, a=a_name)
                            self.list_stack.push(temp_list, temp_lname)
                            self.operand_stack.push(intermediate, new_var_name)
                        elif operator_name in ['[OP_LIST_ODD]', '[OP_LIST_EVEN]']:
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            b = self.intifint(b)
                            a = self.intifint(a)
                            intermediate_list = []
                            self.code_string += "{intermediate_list} = []\n".format(intermediate_list=new_list_name)
                            if operator_name == '[OP_LIST_ODD]':
                                if a%2==0:
                                    for i in range(a+1, b+1, 2):
                                        intermediate_list.append(i)
                                else:
                                    for i in range(a, b+1, 2):
                                        intermediate_list.append(i)
                                self.code_string += "if {a}%2==0:\n".format(a=a_name)
                            elif operator_name == '[OP_LIST_EVEN]':
                                if a%2!=0:
                                    for i in range(a+1, b+1, 2):
                                        intermediate_list.append(i)
                                else:
                                    for i in range(a, b+1, 2):
                                        intermediate_list.append(i)
                                self.code_string += "if {a}%2!=0:\n".format(a=a_name)

                            self.code_string += "    for i in range({a}+1, {b}+1, 2):\n\
        {intermediate_list}.append(i)\n\
else:\n\
    for i in range({a}, {b}+1, 2):\n\
        {intermediate_list}.append(i)\n".format(intermediate_list=new_list_name, a=a_name, b=b_name)

                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_ADD]':
                            temp_list2, temp_lname2 = self.list_stack.pop()
                            temp_list1, temp_lname1 = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            for i in range(len(temp_list1)):
                                intermediate_list.append(self.to_float(temp_list1[i]) + self.to_float(temp_list2[i]))
                            self.code_string += '{intermediate_list}=[]\n\
for i in range(len({temp_list1})):\n\
    {intermediate_list}.append(float({temp_list1}[i])+float({temp_list2}[i]))\n'.format(intermediate_list=new_list_name, temp_list1=temp_lname1, temp_list2=temp_lname2)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_SUB]':
                            temp_list2, temp_lname2 = self.list_stack.pop()
                            temp_list1, temp_lname1 = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            for i in range(len(temp_list1)):
                                intermediate_list.append(self.to_float(temp_list1[i]) - self.to_float(temp_list2[i]))
                            self.code_string += '{intermediate_list}=[]\n\
for i in range(len({temp_list1})):\n\
    {intermediate_list}.append(float({temp_list1}[i])-float({temp_list2}[i]))\n'.format(intermediate_list=new_list_name, temp_list1=temp_lname1, temp_list2=temp_lname2)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_MUL]':
                            temp_list2, temp_lname2 = self.list_stack.pop()
                            temp_list1, temp_lname1 = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            for i in range(len(temp_list1)):
                                intermediate_list.append(self.to_float(temp_list1[i]) * self.to_float(temp_list2[i]))
                            self.code_string += '{intermediate_list}=[]\n\
for i in range(len({temp_list1})):\n\
    {intermediate_list}.append(float({temp_list1}[i])*float({temp_list2}[i]))\n'.format(intermediate_list=new_list_name, temp_list1=temp_lname1, temp_list2=temp_lname2)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_DIV]':
                            temp_list2, temp_lname2 = self.list_stack.pop()
                            temp_list1, temp_lname1 = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            for i in range(len(temp_list1)):
                                intermediate_list.append(self.to_float(temp_list1[i]) / self.to_float(temp_list2[i]))
                            self.code_string += '{intermediate_list}=[]\n\
for i in range(len({temp_list1})):\n\
    {intermediate_list}.append(float({temp_list1}[i])/float({temp_list2}[i]))\n'.format(intermediate_list=new_list_name, temp_list1=temp_lname1, temp_list2=temp_lname2)
                            self.list_stack.push(intermediate_list, new_list_name)
 
                    elif operator_info[1]==3:
                        if operator_name == '[OP_LIST_ARANGE]':
                            c, c_name = self.operand_stack.pop()
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            c = self.intifint(c)
                            b = self.intifint(b)
                            a = self.intifint(a)
                            list_name = self.list_names.pop(0)
                            intermediate_list = [i for i in range(a, b + 1, c)]
                            self.code_string += '{} = [i for i in range({}, {} + 1, {})]\n'.format(list_name, a_name, b_name, c_name)
                            self.list_stack.push(intermediate_list, list_name)
                        elif operator_name == '[OP_LIST_FIND_UNK]':
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            temp_list, temp_lname = self.list_stack.pop()
                            a = str(a)
                            b = str(b)
                            unk_idx = a.index(b)
                            intermediate = []
                            for elem in temp_list:
                                elem = str(elem)
                                intermediate.append(int(elem[unk_idx]))
                            intermediate = list(set(intermediate))
                            if len(intermediate) == 1:
                                intermediate = intermediate[0]

                            if isinstance(intermediate, list):
                                new_list_name = self.list_names.pop(0)
                                self.list_stack.push(temp_list, temp_lname)
                                self.list_stack.push(intermediate, new_list_name)
                                self.code_string += '{a} = str({a})\n\
{b} = str({b})\n\
unk_idx = {a}.index({b})\n\
{intermediate_list} = []\n\
for elem in {temp_list}:\n\
    elem = str(elem)\n\
    {intermediate_list}.append(int(elem[unk_idx]))\n\
{intermediate_list} = list(set({intermediate_list}))\n'.format(a=a_name, b=b_name, intermediate_list=new_list_name, temp_list=temp_lname)
                            else:
                                new_var_name = self.operand_names.pop(0)
                                self.list_stack.push(temp_list, temp_lname)
                                self.operand_stack.push(intermediate, new_var_name)
                                self.code_string += '{a} = str({a})\n\
{b} = str({b})\n\
unk_idx = {a}.index({b})\n\
{intermediate} = 0\n\
for elem in {temp_list}:\n\
    elem = str(elem)\n\
    {intermediate} = int(elem[unk_idx])\n'.format(a=a_name, b=b_name, intermediate=new_var_name, temp_list=temp_lname)
                        elif operator_name == '[OP_LIST_DIVIDE_AND_REMAIN]':
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            b = self.intifint(b)
                            a = self.intifint(a)
                            temp_list, temp_lname = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            a = int(a)
                            b = int(b)
                            if b < 0:
                                b = b + a
                            for i in temp_list:
                                i = int(i)
                                if i%a == b:
                                    intermediate_list.append(i)
                            #print('intermediate_list', intermediate_list)
                            self.code_string += "{intermediate_list} = [] \n\
{a} = int({a})\n\
{b} = int({b})\n\
if {b} < 0:\n\
    {b} = {b} + {a}\n\
for i in {temp_list}:\n\
    i = int(i)\n\
    if i%{a} == {b}:\n\
        {intermediate_list}.append(i)\n".format(intermediate_list=new_list_name, a=a_name, b=b_name, temp_list=temp_lname)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_SEARCH_FIXED_DIGIT]':
                            b, b_name = self.operand_stack.pop()
                            a, a_name = self.operand_stack.pop()
                            b = self.intifint(b)
                            a = self.intifint(a)
                            temp_list, temp_lname = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)
                            intermediate_list = []
                            a = int(a)
                            b = int(b)
                            for i in temp_list:
                                i = int(i)
                                if (i // a) % 10 == b:
                                    intermediate_list.append(i)
                            self.code_string += "{intermediate_list} = [] \n\
{a} = int({a})\n\
{b} = int({b})\n\
for i in {temp_list}:\n\
    i = int(i)\n\
    if (i//{a})%10 == {b}:\n\
        {intermediate_list}.append(i)\n".format(intermediate_list=new_list_name, a=a_name, b=b_name, temp_list=temp_lname)
                            self.list_stack.push(intermediate_list, new_list_name)
                        elif operator_name == '[OP_LIST_COND_BIG_SMALL]':
                            target_list, target_name = self.list_stack.pop()
                            condition_list, condition_name = self.list_stack.pop()
                            entity_list, entity_name = self.list_stack.pop()
                            new_list_name = self.list_names.pop(0)

                            from queue import Queue

                            input_dict = {i: cnt for cnt, i in enumerate(entity_list)}

                            adj_mat = [[0 for _ in range(len(entity_list))] for _ in range(len(entity_list))]
                            is_visited = [[0 for _ in range(len(entity_list))] for _ in range(len(entity_list))]

                            que = Queue()

                            iterate_num = len(condition_list)//3

                            for i in range(iterate_num):
                                operand_1 = condition_list[i*3]
                                operand_2 = condition_list[i*3 + 1]

                                operand_1_id = input_dict[operand_1]
                                operand_2_id = input_dict[operand_2]
                                
                                operator = condition_list[i*3+2]

                                if operator == '>':
                                    adj_mat[operand_1_id][operand_2_id] = 1
                                    is_visited[operand_1_id][operand_2_id] = 1
                                    que.put((operand_1_id, operand_2_id))

                                    adj_mat[operand_2_id][operand_1_id] = -1
                                    is_visited[operand_2_id][operand_1_id] = 1
                                    que.put((operand_2_id, operand_1_id))
                                    
                                elif operator == '<':
                                    adj_mat[operand_1_id, operand_2_id] = -1
                                    is_visited[operand_1_id][operand_2_id] = 1
                                    que.put((operand_1_id, operand_2_id))

                                    adj_mat[operand_2_id, operand_1_id] = 1
                                    is_visited[operand_2_id][operand_1_id] = 1
                                    que.put((operand_2_id, operand_1_id))

                            while not que.empty():
                                operand_1, operand_2 = que.get()

                                if adj_mat[operand_1][operand_2] == 1:
                                    for i in range(0, len(entity_list)):
                                        if (adj_mat[operand_1][i] == -1) and (not is_visited[operand_2][i]):
                                            adj_mat[operand_2][i] = -1
                                            adj_mat[i][operand_2] = 1
                                            is_visited[operand_2][i] = 1
                                            is_visited[i][operand_2] = 1
                                            que.put((operand_2, i))
                                            que.put((i, operand_2))
                                    
                                    for i in range(0, len(entity_list)):
                                        if (adj_mat[operand_2][i] == 1) and (not is_visited[operand_1][i]):
                                            adj_mat[operand_1][i] = 1
                                            adj_mat[i][operand_1] = -1
                                            is_visited[operand_1][i] = 1
                                            is_visited[i][operand_1] = 1
                                            que.put((operand_1, i))
                                            que.put((i, operand_1))

                                if adj_mat[operand_1][operand_2] == -1:
                                    for i in range(0, len(entity_list)):
                                        if (adj_mat[operand_1][i] == 1) and (not is_visited[i][operand_2]):
                                            adj_mat[i][operand_2] = -1
                                            adj_mat[operand_2][i] = 1
                                            is_visited[i][operand_2] = 1
                                            is_visited[operand_2][i] = 1
                                            que.put((i, operand_2))
                                            que.put((operand_2, i))
                                    
                                    for i in range(0, len(entity_list)):
                                        if (adj_mat[operand_2][i] == -1) and (not is_visited[operand_1][i]):
                                            adj_mat[operand_1][i] = -1
                                            adj_mat[i][operand_1] = 1
                                            is_visited[i][operand_1] = 1
                                            is_visited[operand_1][i] = 1
                                            que.put(operand_1, i)
                                            que.put(i, operand_1)

                            operand_1 = target_list[0]
                            operand_2 = target_list[1]

                            operand_1_id = input_dict[operand_1]
                            operand_2_id = input_dict[operand_2]

                            if adj_mat[operand_1_id][operand_2_id] == -1:
                                intermediate_list = [operand_2, operand_1]
                            else:
                                intermediate_list = [operand_1, operand_2]

                            self.list_stack.push(intermediate_list, new_list_name)

                            self.code_string += "from queue import Queue\n\
input_dict = dict()\n\
for cnt, i in enumerate({entity_list}):\n\
    input_dict[i] = cnt\n\
adj_mat = []\n\
for _ in range(len({entity_list})):\n\
    temp_list = []\n\
    for _ in range(len({entity_list})):\n\
        temp_list.append(0)\n\
    adj_mat.append(temp_list)\n\
is_visited = []\n\
for _ in range(len({entity_list})):\n\
    temp_list = []\n\
    for _ in range(len({entity_list})):\n\
        temp_list.append(0)\n\
    is_visited.append(temp_list)\n\
que = Queue()\n\
iterate_num = len({condition})//3\n\
for i in range(iterate_num):\n\
    operand_1 = {condition}[i*3]\n\
    operand_2 = {condition}[i*3 + 1]\n\
    operand_1_id = input_dict[operand_1]\n\
    operand_2_id = input_dict[operand_2]\n\
    operator = {condition}[i*3+2]\n\
    if operator == '>':\n\
        adj_mat[operand_1_id][operand_2_id] = 1\n\
        is_visited[operand_1_id][operand_2_id] = 1\n\
        que.put((operand_1_id, operand_2_id))\n\
        adj_mat[operand_2_id][operand_1_id] = -1\n\
        is_visited[operand_2_id][operand_1_id] = 1\n\
        que.put((operand_2_id, operand_1_id))\n\
    elif operator == '<':\n\
        adj_mat[operand_1_id, operand_2_id] = -1\n\
        is_visited[operand_1_id][operand_2_id] = 1\n\
        que.put((operand_1_id, operand_2_id))\n\
        adj_mat[operand_2_id, operand_1_id] = 1\n\
        is_visited[operand_2_id][operand_1_id] = 1\n\
        que.put((operand_2_id, operand_1_id))\n\
while not que.empty():\n\
    operand_1, operand_2 = que.get()\n\
    if adj_mat[operand_1][operand_2] == 1:\n\
        for i in range(0, len({entity_list})):\n\
            if (adj_mat[operand_1][i] == -1) and (not is_visited[operand_2][i]):\n\
                adj_mat[operand_2][i] = -1\n\
                adj_mat[i][operand_2] = 1\n\
                is_visited[operand_2][i] = 1\n\
                is_visited[i][operand_2] = 1\n\
                que.put((operand_2, i))\n\
                que.put((i, operand_2))\n\
        for i in range(0, len({entity_list})):\n\
            if (adj_mat[operand_2][i] == 1) and (not is_visited[operand_1][i]):\n\
                adj_mat[operand_1][i] = 1\n\
                adj_mat[i][operand_1] = -1\n\
                is_visited[operand_1][i] = 1\n\
                is_visited[i][operand_1] = 1\n\
                que.put((operand_1, i))\n\
                que.put((i, operand_1))\n\
    if adj_mat[operand_1][operand_2] == -1:\n\
        for i in range(0, len({entity_list})):\n\
            if (adj_mat[operand_1][i] == 1) and (not is_visited[i][operand_2]):\n\
                adj_mat[i][operand_2] = -1\n\
                adj_mat[operand_2][i] = 1\n\
                is_visited[i][operand_2] = 1\n\
                is_visited[operand_2][i] = 1\n\
                que.put((i, operand_2))\n\
                que.put((operand_2, i))\n\
        for i in range(0, len({entity_list})):\n\
            if (adj_mat[operand_2][i] == -1) and (not is_visited[operand_1][i]):\n\
                adj_mat[operand_1][i] = -1\n\
                adj_mat[i][operand_1] = 1\n\
                is_visited[i][operand_1] = 1\n\
                is_visited[operand_1][i] = 1\n\
                que.put(operand_1, i)\n\
                que.put(i, operand_1)\n\
operand_1 = {target}[0]\n\
operand_2 = {target}[1]\n\
operand_1_id = input_dict[operand_1]\n\
operand_2_id = input_dict[operand_2]\n\
if adj_mat[operand_1_id][operand_2_id] == -1:\n\
    {intermediate_list} = [operand_2, operand_1]\n\
else:\n\
    {intermediate_list}  = [operand_2, operand_1]\n".format(entity_list=entity_name, target=target_name, condition=condition_name, intermediate_list=new_list_name)
                    else:
                        print("not defined")
                    
            else: # if operand - scalar value
                # var_name = self.operand_names.pop(0)
                # self.operand_stack.push(i, var_name)

                if n + 1 < len(operand_operator_list) and operand_operator_list[n + 1] == "[OP_MEM]":
                    self.operand_stack.push(i, None)  # 변수에 할당하지 않음.
                    continue

                var_name = self.operand_names.pop(0)

                if i in self.mem:
                    self.operand_stack.push(self.mem[i][1], var_name)
                    self.code_string += f"{var_name} = {i}\n"
                    continue

                if self.is_number(i):
                    i = self.to_float(i)
                    if i == int(i):
                        i = int(i)
                    self.code_string += '{} = {}\n'.format(var_name, i)
                    #if self.is_fraction(i):
                    #    self.code_string += '{var} = round({var}+1e-10, 2)\n'.format(var=var_name)
                    #    i = round(self.to_float(i)+1e-10, 2)
                else:
                    self.code_string += "{} = '{}'\n".format(var_name, i)
                
                self.operand_stack.push(i, var_name)


        result, name = self.operand_stack.pop()
        loc = {}
        # print(self.code_string)
        exec(self.code_string, globals(), loc)
        result = loc[name]
        
        # 문제에 분수가 포함될 수도 있으나, 1단계 대회에서는 분수로 답을 하는 경우는 없습니다.
        # 문제정의서에 '정답은 정수로 기재하되 문제에서 소수점의 답을 요구할 경우 반올림하여 소수점 둘째자리까지 기재'로 되어있습니다. 
        # 따라서 소수점의 답을 요구하지 않을 경우 정수로 기재하고 소수점을 요구할 경우 반올림하여 소수점 둘째짜리까지 기재하시면 됩니다. 
        # 예를 들어 문제에서 소수점을 포함하여 답을 하라라고 했을 때 정답이 6.127이면 6.13으로 기재하고 정답이 3.2이면 3.20으로 기재하시면 됩니다.
        try:
            if int(result) != self.to_float(result): # float
                result = '{:.2f}'.format(round(result+1e-10, 2))
                if str(result)[-3:] == ".00":
                    result = int(result[:-3])
                    self.code_string += "print(int(eval('{:.2f}'.format(round(%s+1e-10,2)))))"%name
                else:
                    self.code_string += "print('{:.2f}'.format(round(%s+1e-10,2)))"% name 
            else: # int
                result = int(result)
                self.code_string += 'print(int({}))'.format(name)
        except: # string
            name = name.replace('(', '')
            name = name.replace(')', '')
            self.code_string += 'print({})'.format(name)

        return result, self.code_string