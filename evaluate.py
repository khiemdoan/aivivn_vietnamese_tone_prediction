from predict import predict_sentence
from vietnamese_utils import remove_vietnamese_tone


def evaluate(sentence):
    no_tone_sentence = remove_vietnamese_tone(sentence)
    predicted_sentence = predict_sentence(no_tone_sentence)

    print(f'{sentence} -> {predicted_sentence}')

    original_words = sentence.split()
    predicted_words = predicted_sentence.split()

    correct = 0
    for w1, w2 in zip(original_words, predicted_words):
        if w1 == w2:
            correct += 1
    return correct, len(original_words)


with open('data/mini_train.txt', 'r', encoding='utf-8') as infile:
    lines = infile.readlines()
    lines = [line.strip() for line in lines]


if __name__ == '__main__':
    total_correct = 0
    total_words = 0
    for i, line in enumerate(lines, 1):
        try:
            correct, total = evaluate(line)
            total_correct += correct
            total_words += total
            print(f'{i}: {total_correct / total_words:0.4f} - {line}')
        except Exception as ex:
            print(line)
            print(ex)
