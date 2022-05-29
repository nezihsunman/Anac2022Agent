from collections import defaultdict

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value


class OpponentModel:
    def __init__(self, domain: Domain):
        self.offers = []
        self.domain = domain
        self.offers_unique = []

    def update(self, bid: Bid):
        # keep track of all bids received
        self.offers.append(bid)

        if bid not in self.offers_unique:
            self.offers_unique.append(bid)

