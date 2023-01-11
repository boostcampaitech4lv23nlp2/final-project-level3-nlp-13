import argparse

from crawlers.naver_crawler import NaverCrawler
from credentials import naver_account as headers


def main(args):
    if args.crawler == "naver":
        query = args.query
        headers.update(
            {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
        )
        naver_crawler = NaverCrawler(headers)
        articles = naver_crawler(query=query, n=args.num)
        print(f"Crawled {len(articles)} articles from the given query '{query}'")  # TO-DO: change to logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawler",
        "-c",
        type=str,
        chocies=["naver", "twitter", "theqoo"],
        help="the type of crawler to use",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
    )
    parser.add_argument(
        "--num",
        "-n",
        default=100,
        type=int,
        help="the number of news articles to retrieve for the query",
    )
    args, _ = parser.parse_known_args()
    main(args)
