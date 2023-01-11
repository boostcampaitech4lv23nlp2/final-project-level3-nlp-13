from crawlers.naver_crawler import NaverCrawler
from credentials import naver_account as headers


def main():
    naver_crawler = NaverCrawler(headers)
    outs = naver_crawler(query="bts", n=2)
    print(outs[0])


if __name__ == "__main__":
    main()
