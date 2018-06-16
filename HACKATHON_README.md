# Инструкция по настройке

1. Создайте виртуальную среду Python 3.6. Например, так:
```
virtualenv env
```
Но лучше conda:
```
conda create -n myenv python==3.6
```
2. Активируйте среду:
```
source ./env/bin/activate
```
Или, если это conda:
```
source activate myenv
```
3. Склонируйте ветку `odqa_hack` проекта **DeepPavlov**:
```
git clone -b odqa_hack https://github.com/deepmipt/DeepPavlov.git
```
4. Перейдите в корень проекта:
```
cd DeepPavlov
```
5. Установите нужные зависимости:
```
python setup.py
python -m spacy download en
```
6. Перейдите в deeppavlov:
```
cd deepapvlov/
```
7. Скачайте нужные данные с сервера:
```
python deep.py download deeppavlov/configs/odqa/odqa_hack.json
```
8. Запустите процесс интерактивного общения c ODQA (**stdin-stdout**):
```
python hack.py
```
9. Задавайте ODQA вопросы на русском языке и наслаждайтесь жизнью.

Если что-то не получается, не отчаивайтесь и обратитесь к тому, кто это напрограммировал.