from crawlers.naver_crawler import NaverCrawler
from credentials import naver_account as headers


def main():
    headers.update(
        {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    )
    naver_crawler = NaverCrawler(headers)
    outs = naver_crawler(query="bts", n=2)
    print(outs[0])


if __name__ == "__main__":
    main()
