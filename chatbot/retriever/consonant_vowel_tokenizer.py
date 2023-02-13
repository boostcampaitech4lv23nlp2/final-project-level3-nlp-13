# 한국어 초성/중성/종성 단위 tokenizer


class ConsonantVowelTokenizer:
    def __init__(self):
        # 유니코드의 인덱스에 따라서 초성/중성/종성 리스트를 선언
        # 종성이 없는 경우에는 *를 사용
        self.NO_JONGSUNG = "*"
        # fmt: off
        self.CHOSUNGS = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
        'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.JOONGSUNGS = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.JONGSUNGS = [self.NO_JONGSUNG,  'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ',
        'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        # fmt: on
        self.N_CHOSUNGS = 19
        self.N_JOONGSUNGS = 21
        self.N_JONGSUNGS = 28

        self.FIRST_HANGUL = 0xAC00  #'가'
        self.LAST_HANGUL = 0xD7A3  #'힣'

    def tokenize(self, text):
        # 한글 문자의 유니코드 값 : ( ( 초성 * 21 ) + 중성 ) * 28 + 종성 + 0xAC00
        tokens = []
        for c in text:
            if self.FIRST_HANGUL <= ord(c) <= self.LAST_HANGUL:
                code = ord(c) - self.FIRST_HANGUL
                jong_idx = code % self.N_JONGSUNGS  # 종성
                code = code // self.N_JONGSUNGS
                joong_idx = code % self.N_JOONGSUNGS  # 중성
                code = code // self.N_JOONGSUNGS
                cho_idx = code  # 초성

                tokens.append(self.CHOSUNGS[cho_idx])
                tokens.append(self.JOONGSUNGS[joong_idx])
                tokens.append(self.JONGSUNGS[jong_idx])
            else:
                tokens.append(c)
        return tokens


if __name__ == "__main__":
    tokenizer = ConsonantVowelTokenizer()
    print(tokenizer.tokenize("정구기"))
