# MWP-solver-with-pretrained-language-model
기학습된 언어 모델을 활용하여 자연어를 이해할 수 있는 한글 수학 서술형 문제 풀이 모델

본 연구는 인공지능산업원천기술개발 사업으로 2021.07.01.~2021.12.31.(6개월) 동안 진행된 과제입니다.

## 목표
본 연구는 한글 수학 문장형 문제(MWP, Math Word Problem)를 푸는 심층신경망 기반 모델을 개발합니다.  
수학 문장형 문제는 자연어로 구성된 서술형 수학 문제에 대해 풀이과정과 해답을 제시하는 문제를 의미합니다.

## 실행 방법
다음과 같은 코드로 실행할 수 있습니다.
```
python main.py --use_isc True --beam_size 1 --batch_size 32 --seed 951206 --gpu 0 --dataset 'Korean_MWPs'
```

## 데이터

## 실험 결과
