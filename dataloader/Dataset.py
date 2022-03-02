import os
import re
import json

import numpy as np

from tqdm import tqdm
from time import time
from pythonds.basic import Stack

from IPython import embed

string_list = [
    # (가)-(하)
    '(가)', '(나)', '(다)', '(라)', '(라)', '(마)', '(바)', '(사)', '(아)', '(자)', '(차)', '(카)', '(타)', '(파)', '(하)',
    # 문제에 등장하는 '인물'의 이름
    '남준', '석진', '윤기', '호석', '지민', '태형', '정국', '민영', '유정', '은지', '유나', '경수', '미라', '민주', '현지',
    '상민', '윤정', '현정', '예원', '영표', '재선', '승연', '승기', '혜수', '가은', '미애', '효리', '준수', '예림', '찬우', '슬기'
    # 가족 관계를 나타내는 단어
    '손자', '손녀', '조카', '이모', '삼촌', '동생', '누나', '오빠', '아버지', '어머니', '할머니', '할아버지', '엄마', '아빠', '나', '저', '형', '언니',
    # 신체를 나타내는 단어
    '손가락', '발가락', '팔', '다리',
    # 성별을 구분하는 단어
    '암컷', '수컷', '암탉', '수탉', '여학생', '남학생', '여자', '남자',
    # 색을 나타내는 단어
    '흰색', '검은색', '파란색', '노란색', '초록색', '보라색', '노란색', '빨간색', '주황색', '남색', '검정색',
    # 과목을 나타내는 단어
    '영어', '수학', '국어', '사회', '과학', '음악', '미술', '체육',
    # 동물을 나타내는 단어
    '오리', '닭', '토끼', '물고기', '고래', '거위', '달팽이', '개구리', '강아지', '고양이', '비둘기', '병아리', '개', '강아지', '달팽이', '염소', '홍학', '두루미', '꿩', '돼지',
    # 꽃을 나타내는 단어
    '장미', '백합', '튤립', '카네이션', '국화', '화분', '화단', '꽃병',
    # 운동 관련 단어
    '배구공', '농구공', '축구공', '탁구공', '야구공', '줄넘기', '달리기', '수영', '시합',
    # 음식 관련 단어
    '사과', '배', '감', '귤', '포도', '수박', '참외', '딸기', '복숭아', '바나나', '오렌지',
    '토마토', '무', '당근', '오이', '배추', '상추', '양상추', '감자', '양파',
    '사탕', '김밥', '빵', '라면', '과자', '음료수', '주스', '우유', '달걀', '계란',
    # 학습에 필요한 물품을 나타내는 단어
    '연필', '색연필', '지우개', '공책', '도화지', '색종이', '풀', '테이프', '바둑돌', '구슬', '상자', '나무토막', '장난감', '책장', '책꽂이',
    # 일반적인 장소를 나타내는 단어
    '서점', '마트', '문구점', '집', '학교', '수영장', '교실', '도서관', '박물관', '운동장', '주차장', '정류장', '아파트', '농장', '강당', '경찰서', '소방서', '병원', '약국', '공원',
    # 이동수단을 나타내는 단어
    '비행기', '자동차', '트럭', '배', '자전거', '오토바이', '기차', '버스', '엘리베이터',
    # 건물 관련 용어
    '페인트', '벽', '천장', '문', '울타리',
    # 그 외 trainset에서 추가
    '초코우유', '딸기우유', '바나나우유', '커피우유', '흰우유', '우산', '지팡이', '수조', '양동이', '접시', '사과파이',
]


class Dataset:
    def __init__(self, model_name, data_dir, dataset, add_kor_number, testsets, use_ixc, use_iec, use_isc):
        self.model_name = model_name
        self.data_dir = data_dir
        self.data_name = dataset
        self.testsets = testsets

        if 'chall' in self.data_name:
            self.load_data_chall(model_name, data_dir, dataset, add_kor_number, testsets, use_ixc, use_iec, use_isc)

        # For final submission (dataset/problemsheet.json)
        if 'dataset' in self.data_name:
            self.load_data_submit(model_name, data_dir, dataset, add_kor_number, use_ixc, use_iec, use_isc)

    def load_data_chall(self, model_name, data_dir, dataset, add_kor_number, testsets, use_ixc, use_iec, use_isc):
        # read_json
        train_path = os.path.join(data_dir, self.data_name, 'questions_train.json')
        valid_path = os.path.join(data_dir, self.data_name, 'questions_valid.json')
        with open(train_path, 'r', encoding='utf-8-sig') as f:
            train_json = json.load(f)
        with open(valid_path, 'r', encoding='utf-8-sig') as f:
            valid_json = json.load(f)
        test_paths = [os.path.join(data_dir, self.data_name, f'{test_name}.json') for test_name in testsets]
        test_jsons = []
        for test_path in test_paths:
            with open(test_path, 'r', encoding='utf-8-sig') as f:
                test_jsons.append(json.load(f))

        # initializing
        self.idx2question = dict()
        self.idx2solution = dict()
        self.idx2solution = dict()
        self.idx2qtype = dict()
        self.idx2isstring = dict()
        self.idx2INC = dict()
        self.idx2IXC = dict()
        self.idx2IEC = dict()
        self.idx2ISC = dict()
        self.idx2IMQ = dict()
        self.idx2NET = dict()
        self.idx2postfix = dict()
        self.idx2template = dict()

        # TODO: 사람이름, 가나다라,
        self.netvocab2netidx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[OP]': 3}
        self.netidx2netvocab = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[OP]'}
        self.operator2idx = {'[PAD]': 0}
        self.idx2operator = {0: '[PAD]'}
        self.templatetoken2idx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
        self.idx2templatetoken = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
        self.kornum2num = {'하나': 1, '둘': 2, '셋': 3, '넷': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10,
                           '한': 1, '두': 2, '세': 3, '네': 4}
        self.string1_list = [s for s in string_list if len(s) == 1]
        self.string2_list = [s for s in string_list if len(s) == 2]
        self.string3_list = [s for s in string_list if len(s) == 3]
        self.string4_list = [s for s in string_list if len(s) == 4]
        self.string5_list = [s for s in string_list if len(s) == 5]

        def set_values(json, start_idx):
            idxes = []
            for json_idx in json.keys():
                idx = int(json_idx) + start_idx
                idxes.append(idx)
                # question, postfix
                question = json[json_idx]['question']
                postfix = json[json_idx]['equation_op']
                self.idx2question[idx] = question
                self.idx2postfix[idx] = postfix
                self.idx2isstring[idx] = (len(re.sub(r'[0-9\[\]A-Za-z_ ]', '', postfix)) > 0)
                try:
                    qtype = json[json_idx]['qtype']
                except:
                    qtype = '타입미지정'
                self.idx2qtype[idx] = qtype
                try:
                    solution = json[json_idx]['answer'][0]
                    self.idx2solution[idx] = solution
                except:
                    pass

                # Check if value already exists
                if json[json_idx].get('checked') is True:
                    INC = json[json_idx]['INC']
                    IXC = json[json_idx]['IXC']
                    IEC = json[json_idx]['IEC']
                    ISC = json[json_idx]['ISC']
                    IMQ = json[json_idx]['IMQ']
                    NET = json[json_idx]['NET']
                    template = json[json_idx]['template']
                    self.idx2INC[idx] = INC
                    self.idx2IXC[idx] = IXC
                    self.idx2IEC[idx] = IEC
                    self.idx2ISC[idx] = ISC
                    self.idx2IMQ[idx] = IMQ.strip()
                    self.idx2NET[idx] = NET
                    self.idx2template[idx] = template
                    continue
                else:
                    json[json_idx]['checked'] = False

                # 문장 전처리
                new_question = []
                for word in question.strip().split():
                    # 수사가 등장시 숫자 추가
                    if add_kor_number and (word in self.kornum2num.keys()):
                        new_question.append(str(self.kornum2num[word]))
                    new_question.append(word)
                question = ' '.join(new_question)

                # INC, IMQ, IXC, IEC, ISC
                IMQ = ''
                self.idx2INC[idx] = dict()
                num2INC = dict()
                self.idx2IXC[idx] = dict()
                alpha2IXC = dict()
                self.idx2IEC[idx] = dict()
                eq2IEC = dict()
                self.idx2ISC[idx] = dict()
                str2ISC = dict()

                for word in question.split():
                    # 등식이 등장 시 IEC 부여
                    if '=' in word and use_iec:
                        eq = ''
                        for c in word:
                            if c in '1234567890+-*%./-=ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                                eq += c
                            else:  # 88점입니다.
                                break
                        IEC = '[E' + str(len(self.idx2IEC[idx])) + ']'
                        self.idx2IEC[idx][IEC] = eq
                        eq2IEC[eq] = IEC
                        IMQ += IEC + ' '
                        IMQ += word + ' '

                    # 숫자가 등장시 INC 부여
                    elif word[0].isdigit() or ((word[0] == '-' and len(word) != 1) and word[1].isdigit()):
                        num = ''
                        # 1,000원 -> 1000원
                        for c in re.sub('[,]', '', word):
                            if c in '1234567890./-':  # 소수, 분수, 음수 고려
                                num += c
                            else:  # 88점입니다.
                                break
                        INC = '[N' + str(len(self.idx2INC[idx])) + ']'
                        self.idx2INC[idx][INC] = num
                        num2INC[num] = INC
                        IMQ += INC + ' '
                        IMQ += word + ' '

                    # 영어 대문자가 등장시 IXC 부여
                    elif use_ixc and (word[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                        if alpha2IXC.get(word[0]) is not None:
                            IXC = alpha2IXC[word[0]]
                        else:
                            IXC = '[X' + str(len(self.idx2IXC[idx])) + ']'
                            self.idx2IXC[idx][IXC] = word[0]
                            alpha2IXC[word[0]] = IXC
                        IMQ += IXC + ' '
                        IMQ += word + ' '

                    # 정답식과 문제에 특정 문자열이 등장시 ISC 부여
                    # 특정 문자열이 등장시 ISC 부여
                    elif use_isc and ((re.sub('[,을를이가은는로]', '', word) in self.string1_list) or (word[:2] in self.string2_list) or (word[:3] in self.string3_list) or (word[:4] in self.string4_list) or (word[:5] in self.string5_list)):
                        tmp_str = ''
                        if word[:5] in self.string5_list:
                            tmp_str = word[:5]
                        elif word[:4] in self.string4_list:
                            tmp_str = word[:4]
                        elif word[:3] in self.string3_list:
                            tmp_str = word[:3]
                        elif word[:2] in self.string2_list:
                            tmp_str = word[:2]
                        elif re.sub('[,을를이가은는로]', '', word) in self.string1_list:
                            tmp_str = re.sub('[,을를이가은는로]', '', word)

                        if str2ISC.get(tmp_str) is not None:
                            ISC = str2ISC[tmp_str]
                        else:
                            ISC = '[S' + str(len(self.idx2ISC[idx])) + ']'
                            self.idx2ISC[idx][ISC] = tmp_str
                            str2ISC[tmp_str] = ISC
                        IMQ += ISC + ' '
                        IMQ += word + ' '

                    else:
                        IMQ += word + ' '

                self.idx2IMQ[idx] = IMQ.strip()

                # postfix -> NET (For TM-generation)
                NET = postfix.split()
                # Number -> INC
                for k, v in self.idx2INC[idx].items():
                    for NET_idx, token in enumerate(NET):
                        if v == token:
                            NET[NET_idx] = k
                # 미지수 -> IXC
                for k, v in self.idx2IXC[idx].items():
                    for NET_idx, token in enumerate(NET):
                        if v == token:
                            NET[NET_idx] = k
                # 등식 -> IEC
                for k, v in self.idx2IEC[idx].items():
                    for NET_idx, token in enumerate(NET):
                        if v == token:
                            NET[NET_idx] = k
                # 문자열 -> ISC
                for k, v in self.idx2ISC[idx].items():
                    for NET_idx, token in enumerate(NET):
                        if v == token:
                            NET[NET_idx] = k
                # Constant -> C
                for NET_idx, token in enumerate(NET):
                    if token[0].isdigit() or (token[0] == '-' and token[1].isdigit()) or token in '><':
                        NET[NET_idx] = '[C' + token + ']'
                # Operation -> OP & Constant 처리
                for NET_idx, token in enumerate(NET):
                    if token.startswith('[OP'):
                        if self.operator2idx.get(token) is None:
                            self.operator2idx[token] = len(self.operator2idx)
                            self.idx2operator[self.operator2idx[token]] = token
                        NET[NET_idx] = '[OP]'
                    else:
                        if self.netvocab2netidx.get(token) is None:
                            self.netvocab2netidx[token] = len(self.netvocab2netidx)
                            self.netidx2netvocab[self.netvocab2netidx[token]] = token
                # for NET_idx, token in enumerate(NET):
                #     if self.netvocab2netidx.get(token) is None:
                #         self.netvocab2netidx[token] = len(self.netvocab2netidx)
                #         self.netidx2netvocab[self.netvocab2netidx[token]] = token
                #     if token.startswith('[OP'):
                #         if self.operator2idx.get(token) is None:
                #             self.operator2idx[token] = len(self.operator2idx)
                #             self.idx2operator[self.operator2idx[token]] = token
                #         NET[NET_idx] = token
                self.idx2NET[idx] = ' '.join(NET)

                # postfix -> template (For GEO)
                template = postfix.split()
                for k, v in self.idx2INC[idx].items():
                    for template_idx, token in enumerate(template):
                        if v == token:
                            template[template_idx] = k
                # 미지수 -> IXC
                for k, v in self.idx2IXC[idx].items():
                    for template_idx, token in enumerate(template):
                        if v == token:
                            template[template_idx] = k
                # 등식 -> IEC
                for k, v in self.idx2IEC[idx].items():
                    for template_idx, token in enumerate(template):
                        if v == token:
                            template[template_idx] = k
                # 문자열 -> ISC
                for k, v in self.idx2ISC[idx].items():
                    for template_idx, token in enumerate(template):
                        if v == token:
                            template[template_idx] = k
                # Constant -> C
                for template_idx, token in enumerate(template):
                    if token[0].isdigit() or (token[0] == '-' and token[1].isdigit()) or token in '><':
                        template[template_idx] = '[C' + token + ']'
                # templatetoken dict에 추가
                for template_idx, token in enumerate(template):
                    if self.templatetoken2idx.get(token) is None:
                        self.templatetoken2idx[token] = len(self.templatetoken2idx)
                        self.idx2templatetoken[self.templatetoken2idx[token]] = token
                self.idx2template[idx] = ' '.join(template)
            return np.array(idxes)

        # Set train/valid/test ids
        self.train_ids = set_values(train_json, start_idx=0)
        self.valid_ids = set_values(valid_json, start_idx=1000000)
        self.test_ids = []
        for i, test_json in enumerate(test_jsons):
            test_ids = set_values(test_json, start_idx=10000000*(i+1))
            self.test_ids.append(test_ids)

        # Set question type ids
        self.idx2qtype_id = dict()
        map_qtype_id = dict()
        for idx, qtype in self.idx2qtype.items():
            if map_qtype_id.get(qtype) is None:
                map_qtype_id[qtype] = len(map_qtype_id)
            self.idx2qtype_id[idx] = map_qtype_id[qtype]

        # save file for debugging
        self.save_dataloader_to_file(train_json, train_path, start_idx=0)
        self.save_dataloader_to_file(valid_json, valid_path, start_idx=1000000)
        for i, (test_json, test_path) in enumerate(zip(test_jsons, test_paths)):
            self.save_dataloader_to_file(test_json, test_path, start_idx=10000000*(i+1))

    def load_data_submit(self, model_name, data_dir, dataset, add_kor_number, use_ixc, use_iec, use_isc):
        # read_json (dataset/problemsheet.json)
        test_path = os.path.join(self.data_name, 'problemsheet_5_00.json')
        with open(test_path, 'r', encoding='utf-8-sig') as f:
            test_json = json.load(f)

        # initializing
        self.idx2question = dict()
        self.idx2INC = dict()
        self.idx2IXC = dict()
        self.idx2IEC = dict()
        self.idx2ISC = dict()
        self.idx2IMQ = dict()

        # TODO: 사람이름, 가나다라, A, B, C, D, ... 고려
        self.netvocab2netidx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[OP]': 3}
        self.netidx2netvocab = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[OP]'}
        self.operator2idx = {'[PAD]': 0}
        self.idx2operator = {0: '[PAD]'}
        self.templatetoken2idx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
        self.idx2templatetoken = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
        self.kornum2num = {'하나': 1, '둘': 2, '셋': 3, '넷': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10,
                           '한': 1, '두': 2, '세': 3, '네': 4}
        self.string1_list = [s for s in string_list if len(s) == 1]
        self.string2_list = [s for s in string_list if len(s) == 2]
        self.string3_list = [s for s in string_list if len(s) == 3]
        self.string4_list = [s for s in string_list if len(s) == 4]
        self.string5_list = [s for s in string_list if len(s) == 5]

        def set_values(json, start_idx):
            idxes = []
            for json_idx in json.keys():
                idx = int(json_idx) + start_idx
                idxes.append(idx)
                # question
                question = json[json_idx]['question']
                self.idx2question[idx] = question

                # 문장 전처리
                new_question = []
                for word in question.strip().split():
                    # 수사가 등장시 숫자 추가
                    if add_kor_number and (word in self.kornum2num.keys()):
                        new_question.append(str(self.kornum2num[word]))
                    new_question.append(word)
                question = ' '.join(new_question)

                # INC, IMQ, IXC, IEC, ISC
                IMQ = ''
                self.idx2INC[idx] = dict()
                num2INC = dict()
                self.idx2IXC[idx] = dict()
                alpha2IXC = dict()
                self.idx2IEC[idx] = dict()
                eq2IEC = dict()
                self.idx2ISC[idx] = dict()
                str2ISC = dict()

                for word in question.split():
                    # 등식이 등장 시 IEC 부여
                    if '=' in word and use_iec:
                        eq = ''
                        for c in word:
                            if c in '1234567890+-*%./-=ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                                eq += c
                            else:  # 88점입니다.
                                break
                        IEC = '[E' + str(len(self.idx2IEC[idx])) + ']'
                        self.idx2IEC[idx][IEC] = eq
                        eq2IEC[eq] = IEC
                        IMQ += IEC + ' '
                        IMQ += word + ' '

                    # 숫자가 등장시 INC 부여
                    elif word[0].isdigit() or ((word[0] == '-' and len(word) != 1) and word[1].isdigit()):
                        num = ''
                        # 1,000원 -> 1000원
                        for c in re.sub('[,]', '', word):
                            if c in '1234567890./-':  # 소수, 분수, 음수 고려
                                num += c
                            else:  # 88점입니다.
                                break
                        INC = '[N' + str(len(self.idx2INC[idx])) + ']'
                        self.idx2INC[idx][INC] = num
                        num2INC[num] = INC
                        IMQ += INC + ' '
                        IMQ += word + ' '

                    # 영어 대문자가 등장시 IXC 부여
                    elif use_ixc and (word[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                        if alpha2IXC.get(word[0]) is not None:
                            IXC = alpha2IXC[word[0]]
                        else:
                            IXC = '[X' + str(len(self.idx2IXC[idx])) + ']'
                            self.idx2IXC[idx][IXC] = word[0]
                            alpha2IXC[word[0]] = IXC
                        IMQ += IXC + ' '
                        IMQ += word + ' '

                    # 특정 문자열이 등장시 ISC 부여
                    elif use_isc and ((re.sub('[,을를이가은는로]', '', word) in self.string1_list) or (word[:2] in self.string2_list) or (word[:3] in self.string3_list) or (word[:4] in self.string4_list) or (word[:5] in self.string5_list)):
                        tmp_str = ''
                        if word[:5] in self.string5_list:
                            tmp_str = word[:5]
                        elif word[:4] in self.string4_list:
                            tmp_str = word[:4]
                        elif word[:3] in self.string3_list:
                            tmp_str = word[:3]
                        elif word[:2] in self.string2_list:
                            tmp_str = word[:2]
                        elif re.sub('[,을를이가은는로]', '', word) in self.string1_list:
                            tmp_str = re.sub('[,을를이가은는로]', '', word)

                        if str2ISC.get(tmp_str) is not None:
                            ISC = str2ISC[tmp_str]
                        else:
                            ISC = '[S' + str(len(self.idx2ISC[idx])) + ']'
                            self.idx2ISC[idx][ISC] = tmp_str
                            str2ISC[tmp_str] = ISC
                        IMQ += ISC + ' '
                        IMQ += word + ' '
                    else:
                        IMQ += word + ' '

                self.idx2IMQ[idx] = IMQ.strip()
            return np.array(idxes)

        # Set train/valid/test ids
        self.test_ids = set_values(test_json, start_idx=0)

    def load_data_CC(self, model_name, data_dir, dataset):

        # read_json
        data_path = os.path.join(data_dir, self.data_name, 'questions.json')
        with open(data_path, 'r') as f:
            all_json = json.load(f)

        # Set train/valid/test ids
        all_ids = np.arange(len(all_json))
        np.random.shuffle(all_ids)
        self.train_ids = all_ids[:int(0.7 * len(all_ids))]
        self.valid_ids = all_ids[int(0.7 * len(all_ids)): int(0.8 * len(all_ids))]
        self.test_ids = all_ids[int(0.8 * len(all_ids)):]

        # initializing
        self.idx2question = dict()
        self.idx2alignment = dict()
        self.idx2solution = dict()
        self.idx2equation = dict()
        self.idx2INC = dict()
        self.idx2IMQ = dict()
        self.idx2NET = dict()
        self.idx2postfix = dict()

        # TODO: Constant 고려 필요 (예정)
        self.netvocab2netidx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, 'OP': 3}
        self.netidx2netvocab = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: 'OP'}
        self.operator2idx = {'[PAD]': 0}
        self.idx2operator = {0: '[PAD]'}

        # Set Values using json
        for i in range(len(all_json)):
            idx = all_json[i]['iIndex']

            # question, alignment, solution, equation
            question = all_json[i]['sQuestion']
            alignments = all_json[i]['lAlignments']
            solution = all_json[i]['lSolutions'][0]
            equation = all_json[i]['lEquations'][0]
            self.idx2question[idx] = question
            self.idx2alignment[idx] = alignments
            self.idx2solution[idx] = solution
            self.idx2equation[idx] = equation

            # INC, IMQ
            self.idx2INC[idx] = dict()
            IMQ = ''
            num2INC = dict()
            for word in question.split():
                # 숫자가 등장시 INC 부여
                re_word = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', word)
                if re_word.isdigit():
                    INC = 'N' + str(len(self.idx2INC[idx]))
                    self.idx2INC[idx][INC] = re_word
                    num2INC[re_word] = INC
                    IMQ += INC + ' '
                IMQ += word + ' '
            self.idx2IMQ[idx] = IMQ.strip()

            # infix -> postfix
            postfix = self.infixToPostfix(self.make_space_eq(equation[2:]))
            self.idx2postfix[idx] = postfix

            # postfix -> NET
            NET = postfix.split()
            for k, v in self.idx2INC[idx].items():
                for NET_idx, token in enumerate(NET):
                    if v+'.0' == token:
                        NET[NET_idx] = k
                        break
            for NET_idx, token in enumerate(NET):
                if token in '+-*/':
                    if self.operator2idx.get(token) is None:
                        self.operator2idx[token] = len(self.operator2idx)
                        self.idx2operator[self.operator2idx[token]] = token
                    NET[NET_idx] = 'OP'
                else:
                    if self.netvocab2netidx.get(token) is None:
                        self.netvocab2netidx[token] = len(self.netvocab2netidx)
                        self.netidx2netvocab[self.netvocab2netidx[token]] = token
            self.idx2NET[idx] = ' '.join(NET)

    def infixToPostfix(self, infixexpr):
        prec = {}
        prec["*"] = 3
        prec["/"] = 3
        prec["+"] = 2
        prec["-"] = 2
        prec["("] = 1
        opStack = Stack()
        postfixList = []
        tokenList = infixexpr.split()

        for token in tokenList:
            if re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', token).isdigit():
                postfixList.append(token)
            elif token == '(':
                opStack.push(token)
            elif token == ')':
                topToken = opStack.pop()
                while topToken != '(':
                    postfixList.append(topToken)
                    topToken = opStack.pop()
            else:
                while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
                    postfixList.append(opStack.pop())
                opStack.push(token)

        while not opStack.isEmpty():
            postfixList.append(opStack.pop())
        return " ".join(postfixList)

    def make_space_eq(self, infix):
        new_infix = ''

        for c in infix:
            if c in '+-*/()=%':
                new_infix += ' ' + c + ' '
            else:
                new_infix += c
        return new_infix

    def __str__(self):
        ret_str = '\n'
        ret_str += 'Dataset: %s\n' % self.data_name
        # ret_str += '# of docs_data: %d\n' % len(self.docs_data)
        # ret_str += '# of rels_data: %d(%d+%d)\n' % (self.num_rels_train + self.num_rels_test, self.num_rels_train, self.num_rels_test)
        return ret_str

    def save_dataloader_to_file(self, orig_json, data_path, start_idx=0):
        for json_idx in orig_json.keys():
            idx = int(json_idx)+start_idx
            orig_json[json_idx]['INC'] = self.idx2INC.get(idx)
            orig_json[json_idx]['IXC'] = self.idx2IXC.get(idx)
            orig_json[json_idx]['IEC'] = self.idx2IEC.get(idx)
            orig_json[json_idx]['ISC'] = self.idx2ISC.get(idx)
            orig_json[json_idx]['IMQ'] = self.idx2IMQ.get(idx)
            orig_json[json_idx]['NET'] = self.idx2NET.get(idx)
            orig_json[json_idx]['template'] = self.idx2template.get(idx)

        with open(data_path, 'w', encoding='UTF-8') as f:
            f.write(json.dumps(orig_json, ensure_ascii=False, indent=4))
            print(f"> Successfullly saved processed file at {data_path}")
