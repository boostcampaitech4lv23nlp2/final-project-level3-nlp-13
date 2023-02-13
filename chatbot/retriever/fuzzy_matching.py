import numpy as np
from chatbot.retriever.consonant_vowel_tokenizer import ConsonantVowelTokenizer


class FuzzyMatcher:
    def __init__(self):
        self.tokenizer = ConsonantVowelTokenizer()

    def levenshtein_ratio(self, s, t):
        """
        Calculates levenshtein distance between two strings.
        the function computes the levenshtein distance ratio of similarity between two strings.
        For all i and j, distance[i,j] will contain the Levenshtein distance between the first i characters of s and the first j characters of t.
        """
        s = "".join(self.tokenizer.tokenize(s))
        t = "".join(self.tokenizer.tokenize(t))

        # Initialize matrix of zeros
        rows = len(s) + 1
        cols = len(t) + 1
        distance = np.zeros((rows, cols), dtype=int)

        # 두 입력 string의 자음 모음 tokenzied된 것을 0으로 채워진 2차원 배열 선언
        for i in range(1, rows):
            for k in range(1, cols):
                distance[i][0] = i
                distance[0][k] = k

        # deletion/insertion/substitution cost를 계산
        for col in range(1, cols):
            for row in range(1, rows):
                # substitution cost
                # 해당 idx의 문자가 같으면 cost는 0
                if s[row - 1] == t[col - 1]:
                    cost = 0
                else:
                    # 아니면 2
                    cost = 2
                distance[row][col] = min(
                    distance[row - 1][col] + 1,  # deletion일 경우 cost 1 추가
                    distance[row][col - 1] + 1,  # insertion일 경우 cost 1 추가
                    distance[row - 1][col - 1] + cost,  # substitution일 경우 cost 2 추가
                )

        # Levenshtein Distance Ratio
        Ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return Ratio


if __name__ == "__main__":
    fuzzy_matcher = FuzzyMatcher()
    # Str1 = "정꾹이"
    # Str2 = "정꾸기"
    Str1 = "생일"
    Str2 = "쌩일"

    result = fuzzy_matcher.levenshtein_ratio(Str1, Str2)
    print(result)
