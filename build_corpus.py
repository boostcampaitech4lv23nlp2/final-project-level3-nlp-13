import argparse
import datetime
import pytz

from crawlers import (
    NaverCrawler,
    TwitterCrawler,
    TheqooCrawler,
    KinCrawler,
)


def main(args):
    runtime = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d")

    if args.crawler == "naver":
        query = args.query
        naver_crawler = NaverCrawler(runtime=runtime)
        since, until = args.range.split("~")
        naver_crawler(query=query, n=args.num, since=since, until=until)

    elif args.crawler == "twitter":
        screen_name = args.screen_name
        twitter_crawler = TwitterCrawler()
        twitter_crawler(screen_name=args.screen_name)

    elif args.crawler == "theqoo":
        screen_name = args.screen_name
        theqoo_crawler = TheqooCrawler()
        theqoo_crawler(n=args.num)

    elif args.crawler == "kin":
        query = args.query
        n = args.num
        kin_crawler = KinCrawler(runtime=runtime)
        kin_crawler(query=query, n=n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawler",
        "-c",
        type=str,
        choices=["naver", "twitter", "theqoo", "kin"],
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
        default="~",
        type=str,
        help="YYYY-MM-DD~YYYY-MM-DD. Specify search time range for NaverCrawler",
    )

    parser.add_argument(
        "--screen_name",
        "-s",
        type=str,
        help="screen name of the twitter user to crawl",
    )
    args, _ = parser.parse_known_args()

    main(args)
