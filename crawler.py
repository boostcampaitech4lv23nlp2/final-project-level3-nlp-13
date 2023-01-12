import argparse
import datetime
import pytz

from crawlers.naver_crawler import NaverCrawler
from crawlers.twitter_crawler import TwitterCrawler


def main(args):
    runtime = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%H-%M")

    if args.crawler == "naver":
        query = args.query
        naver_crawler = NaverCrawler(runtime=runtime)
        naver_crawler(query=query, n=args.num)

    elif args.crawler == "twitter":
        screen_name = args.screen_name
        twitter_crawler = TwitterCrawler()
        twitter_crawler(screen_name=args.screen_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawler",
        "-c",
        type=str,
        choices=["naver", "twitter", "theqoo"],
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
    ### twitter crawling args ###
    parser.add_argument(
        "--screen_name",
        "-s",
        type=str,
        help="screen name of the twitter user to crawl",
    )
    args, _ = parser.parse_known_args()
    
    if args.crawler == "naver":
        assert args.num <= 100, "The maximum 'num for NaverCrawler is 100"

    main(args)
