import argparse
import datetime
import pytz
import re
import pandas as pd

from crawlers import (
    NaverCrawler,
    TwitterCrawler,
    TheqooCrawler,
    KinCrawler,
    KinFilter,
    NewsCrawler,
    CommentCrawler,
)


def main(args):
    runtime = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d")

    if args.crawler == "naver":
        naver_crawler = NaverCrawler(runtime=runtime)
        if args.do_crawl:
            query = args.query
            naver_crawler(query=query, n=args.num)
        if args.do_preprocess:
            naver_crawler.preprocess(raw_data_path=args.path)  # 1차 중복 제거: 기사 제목

    elif args.crawler == "twitter":
        screen_name = args.screen_name
        twitter_crawler = TwitterCrawler()
        if args.do_crawl:
            twitter_crawler(screen_name=screen_name)

    elif args.crawler == "theqoo":
        theqoo_crawler = TheqooCrawler()
        if args.do_crawl:
            theqoo_crawler(n=args.num)

    elif args.crawler == "aihub":
        if args.query == "news":
            directory = "data/raw_data/aihub/news" if args.path is None else args.path
            aihub_crawler = NewsCrawler(directory)
            if args.do_crawl:
                aihub_crawler()
            if args.do_preprocess:
                df = pd.read_csv(args.path)
                aihub_crawler.preprocess(df)

        elif args.query == "comment":
            directory = (
                "data/raw_data/aihub/comment" if args.path is None else args.path
            )
            aihub_crawler = CommentCrawler(directory)
            aihub_crawler()

    elif args.crawler == "kin":
        kin_crawler = KinCrawler(runtime=runtime)
        if args.do_crawl:
            query = args.query
            n = args.num
            kin_crawler(query=query, n=n)
        if args.do_preprocess:
            kin_filter = KinFilter() #TO-DO: vocab.json
            df = kin_filter.preprocess(args.path)
            kin_filter.save_csv(df, f"kin_{runtime}.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawler",
        "-c",
        type=str,
        choices=["naver", "twitter", "theqoo", "aihub", "kin"],
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
        help="YYYY.MM.DD~YYYY.MM.DD. Specify search time range for NaverCrawler",
    )

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

    if args.range != "~":
        assert (
            re.match(
                r"[0-9]{4}\.[0-9]{2}\.[0-9]{2}~[0-9]{4}\.[0-9]{2}\.[0-9]{2}", args.range
            ).group()
            == args.range
        ), print("Make sure the given 'range' is formatted correct")

    main(args)
