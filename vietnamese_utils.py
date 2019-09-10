import re
import string

uni_chars_l = 'áàảãạâấầẩẫậăắằẳẵặđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ'
uni_chars_u = 'ÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ'

AEIOUYD = ['a', 'e', 'i', 'o', 'u', 'y', 'd']
A_FAMILY = list('aáàảãạăắằẳẵặâấầẩẫậ')
E_FAMILY = list('eéèẻẽẹêếềểễệ')
I_FAMILY = list('iíìỉĩị')
O_FAMILY = list('oóòỏõọôốồổỗộơớờởỡợ')
U_FAMILY = list('uúùủũụưứừửữự')
Y_FAMILY = list('yýỳỷỹỵ')
D_FAMILY = list('dđ')

tones_l = [
    '1', '2', '3', '4', '5',
    '6', '61', '62', '63', '64', '65',
    '8', '81', '82', '83', '84', '85',
    '9', '1', '2', '3', '4', '5',
    '6', '61', '62', '63', '64', '65',
    '1', '2', '3', '4', '5',
    '1', '2', '3', '4', '5',
    '6', '61', '62', '63', '64', '65',
    '7', '71', '72', '73', '74', '75',
    '1', '2', '3', '4', '5',
    '7', '71', '72', '73', '74', '75',
    '1', '2', '3', '4', '5'
]

tones_u = [
    '1', '2', '3', '4', '5',
    '6', '61', '62', '63', '64', '65',
    '8', '81', '82', '83', '84', '85',
    '9', '1', '2', '3', '4', '5',
    '6', '61', '62', '63', '64', '65',
    '1', '2', '3', '4', '5',
    '1', '2', '3', '4', '5',
    '6', '61', '62', '63', '64', '65',
    '7', '71', '72', '73', '74', '75',
    '1', '2', '3', '4', '5',
    '7', '71', '72', '73', '74', '75',
    '1', '2', '3', '4', '5'
]

no_tone_chars_l = 'a'*17 + 'd' + 'e'*11 + 'i'*5 + 'o'*17 + 'u'*11 + 'y'*5
no_tone_chars_u = 'A'*17 + 'D' + 'E'*11 + 'I'*5 + 'O'*17 + 'U'*11 + 'Y'*5


tones_dict = dict(zip(uni_chars_l + uni_chars_u, tones_l + tones_u))
no_tone_dict = dict(zip(uni_chars_l + uni_chars_u, no_tone_chars_l + no_tone_chars_u))


def decompose_predicted(text: str) -> str:
    text = [c for c in text if c in tones_dict]
    if len(text) == 0:
        return '0'
    text = [tones_dict[c] for c in text]
    return ''.join(text)


def remove_vietnamese_tone(text: str) -> str:
    text = [no_tone_dict.get(c, c) for c in text]
    return ''.join(text)


def is_vietnamese_word(word: str) -> bool:
    word = remove_vietnamese_tone(word)
    word = word.lower()
    pattern = r'^[bcdghklmnpqrstvx]{0,3}[aeiouy]{1,3}[bcdghklmnpqrstvx]{0,2}$'
    return bool(re.match(pattern, word))


def get_vietnamese_alphabet() -> str:
    return string.ascii_lowercase + tones_l
