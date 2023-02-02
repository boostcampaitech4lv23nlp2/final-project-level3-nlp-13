import typing
from dataclasses import dataclass, field, asdict


@dataclass
class Default:

    @property
    def __dict__(self):
        return asdict(self)

@dataclass
class UserTweet(Default):
    user_id: str
    screen_name: str
    message: str

