import argparse
import datetime
import pytz

from crawlers.naver_crawler import NaverCrawler
from crawlers.twitter_crawler import TwitterCrawler
from crawlers.theqoo_crawler import TheqooCrawler


def main(args):
    runtime = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d")

    if args.crawler == "naver":
        query = args.query
        naver_crawler = NaverCrawler(runtime=runtime)
        if args.do_crawl:
            naver_crawler(query=query, n=args.num)
        if args.do_preprocess:
            naver_crawler.preprocess(raw_data_path=args.path)  # 1차 중복 제거: 기사 제목

    elif args.crawler == "twitter":
        screen_name = args.screen_name
        twitter_crawler = TwitterCrawler()
        if args.do_crawl:
            twitter_crawler(screen_name=args.screen_name)

    elif args.crawler == "theqoo":
        screen_name = args.screen_name
        theqoo_crawler = TheqooCrawler()
        if args.do_crawl:
            theqoo_crawler(n=args.num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawler",
        "-c",
        type=str,
        choices=["naver", "twitter", "theqoo"],
        required=True,
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
    parser.add_argument(
        "--range",
        "-r",
        default=" ~ ",
        type=str,
        help="YYYY-MM-DD~YYYY-MM-DD. Specify search time range for NaverCrawler",
    )

    ### twitter crawling args ###
    parser.add_argument(
        "--screen_name",
        "-s",
        type=str,
        help="screen name of the twitter user to crawl",
    )
    # preprocessor
    parser.add_argument(
        "--do_crawl",
        action="store_true",
    )
    parser.add_argument(
        "--do_preprocess",
        action="store_true",
    )
    parser.add_argument(
        "--path", "-p", type=str, default=None, help="path of raw_data to preprocess"
    )
    args, _ = parser.parse_known_args()

    if args.do_crawl:
        assert args.path is None, "--path is preprocessing-only"
    main(args)
