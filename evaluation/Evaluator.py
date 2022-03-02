import os
import re
import json
import numpy as np

from collections import OrderedDict
from pythonds.basic import Stack
from sklearn.metrics import accuracy_score
from IPython import embed
from evaluation.PostfixConverter import PostfixConverter

class Evaluator:
    def __init__(self, early_stop_measure='acc_equation'):
        self.early_stop_measure = early_stop_measure
        self.pf_converter = PostfixConverter()

    def evaluate(self, model, dataset, mode='valid'):
        if mode == 'valid':
            eval_id = dataset.valid_ids
        elif 'test' in mode:
            test_num = dataset.testsets.index(mode)
            eval_id = dataset.test_ids[test_num]
        elif mode == 'submit':
            eval_id = dataset.test_ids

        # Get score
        if mode == 'submit':
            self.get_score_submission(model, dataset, mode, eval_id)
        else:
            score = self.get_score(model, dataset, mode, eval_id)
            return score

    def get_score_submission(self, model, dataset, mode, eval_id):
        # get answer and equation
        eval_answer, eval_equation, _ = model.predict(mode, self.pf_converter)

        equations = []
        eval_equation_answer = []

        for eq in eval_equation:
            try:
                result, code_string = self.pf_converter.convert(eq)
                eval_equation_answer.append(result)
                equations.append(code_string)
            except:
                eval_equation_answer.append(0)
                equations.append('')

        # int, float to .2f
        np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})
        eval_equation_answer = [f"{ans}" for ans in eval_equation_answer]

        answer_dict = {}
        for i, idx in enumerate(eval_id):
            answer_dict[str(idx)] = {}
            answer_dict[str(idx)]['answer'] = eval_equation_answer[i]
            answer_dict[str(idx)]['equation'] = equations[i]

        with open('answersheet_5_00_zxcvxd.json', 'w', encoding='UTF-8') as f:
            f.write(json.dumps(answer_dict, ensure_ascii=False, indent=4))
    

    def get_score(self, model, dataset, mode, eval_id, test_num=None):
        score = OrderedDict()

        true_answer = []
        for idx in eval_id:
            try:
                true_answer.append(self.pf_converter.convert(dataset.idx2postfix[idx])[0])
            except:
                true_answer.append(0)
                print(f"{mode}", idx, dataset.idx2postfix[idx], "is Errored!!!")
        # get answer and equation
        eval_answer, eval_equation, eval_loss = model.predict(mode, self.pf_converter)
        eval_equation_answer = []
        error_list = []
        for eq in eval_equation:
            try: 
                eval_equation_answer.append(self.pf_converter.convert(eq)[0])
                error_list.append(0)
            except:
                eval_equation_answer.append(0)
                error_list.append(1)
        num_error = np.sum(error_list)

        # calculate score
        score[f'{mode}_loss'] = np.mean(eval_loss)
        if eval_answer is not None:
            score['acc_ans'] = accuracy_score(true_answer, eval_answer)
        
        # int, float to .2f
        np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})
        true_answer = [f"{ans}" for ans in true_answer]
        eval_equation_answer = [f"{ans}" for ans in eval_equation_answer]
        score[f'{mode}_accuracy'] = accuracy_score(true_answer, eval_equation_answer)
        score[f'{mode}_num_error'] = num_error
        score[f'{mode}_error_rate'] = num_error / len(eval_equation_answer)

        # calculate score for question type
        type2pred = {type:[] for type in dataset.idx2qtype.values()}
        type2true = {type:[] for type in dataset.idx2qtype.values()}
        for i, idx in enumerate(eval_id):
            type2true[dataset.idx2qtype[idx]].append(true_answer[i])
            type2pred[dataset.idx2qtype[idx]].append(eval_equation_answer[i])

        # Save Predicted
        out_lines = []
        out_lines.append(f"Accuacy: {accuracy_score(true_answer, eval_equation_answer)} ({(np.array(true_answer)==np.array(eval_equation_answer)).sum()}/{len(true_answer)}) , num_error: {num_error}, error_rate: {num_error / len(eval_equation_answer)}")
        out_lines.append(f"\n")
        for type in set(dataset.idx2qtype.values()):
            out_lines.append(f"{type}: -> {accuracy_score(type2true[type], type2pred[type])} ({(np.array(type2true[type])==np.array(type2pred[type])).sum()}/{len(type2pred[type])})")
        out_lines.append(f"\n")
            
        for i, idx in enumerate(eval_id):
            out_lines.append(f"Index: {idx}")
            out_lines.append(f"Question type: {dataset.idx2qtype[idx]}")
            out_lines.append(f"Question: {dataset.idx2question[idx]}")
            out_lines.append(f"True_postfix: {dataset.idx2postfix[idx]} --> {true_answer[i]}")
            out_lines.append(f"Pred_postfix: {eval_equation[i]} --> {eval_equation_answer[i]}")
            out_lines.append(f"Correct: {true_answer[i] == eval_equation_answer[i]}")
            out_lines.append(f"Error: {True if error_list[i] else False}")
            out_lines.append("\n")
        with open(os.path.join(model.log_dir, f'{mode}_Results.txt'),'w') as f:
            f.write('\n'.join(out_lines))

        return score


def eval_postfix(postfixExpr):
    operandStack = Stack()
    tokenList = postfixExpr.split()

    for token in tokenList:
        if re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', token).isdigit():
            operandStack.push(float(token))
        else:
            if operandStack.size() < 2: return 0
            operand2 = operandStack.pop()
            operand1 = operandStack.pop()
            result = doMath(token,operand1,operand2)
            operandStack.push(result)
    return operandStack.pop()

def doMath(op, op1, op2):
    if op == "*":
        return op1 * op2
    elif op == "/":
        return op1 / op2
    elif op == "+":
        return op1 + op2
    else:
        return op1 - op2