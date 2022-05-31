import json
from decimal import Decimal
from typing import Optional

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Value import Value


class ProfileParser:
    def __init__(self):
        self.data = None

    def parse(self, path):
        with open(path) as file:
            self.data = json.load(file)

    def getUtility(self, bid: Bid) -> Decimal:
        return sum([self._util(iss, bid.getValue(iss)) for iss in
                    self.data["LinearAdditiveUtilitySpace"]["issueWeights"].keys()])

    def getUtility_for_testing(self, bid: Bid) -> Decimal:
        return sum([self._util(iss, bid.getValue(iss)) for iss in
                    self.data["LinearAdditiveUtilitySpace"]["issueWeights"].keys()])  # type:ignore

    def _util(self, issue: str, value: Optional[Value]) -> Decimal:
        '''
        @param issue the issue to get weighted util for
        @param value the issue value to use (typically coming from a bid). Can be None
        @return weighted util of just the issue value:
                issueUtilities[issue].utility(value) * issueWeights[issue)
        '''
        if not value:
            return Decimal(0)
        return Decimal(self.data["LinearAdditiveUtilitySpace"]["issueWeights"][issue] * \
                       self.data["LinearAdditiveUtilitySpace"]["issueUtilities"][issue]["DiscreteValueSetUtilities"][
                           "valueUtilities"][str(value)[1:-1]])

    def _util_testing(self, issue: str, value: Optional[Value]) -> Decimal:
        '''
        @param issue the issue to get weighted util for
        @param value the issue value to use (typically coming from a bid). Can be None
        @return weighted util of just the issue value:
                issueUtilities[issue].utility(value) * issueWeights[issue)
        '''
        if not value:
            return Decimal(0)
        return Decimal(self.data["LinearAdditiveUtilitySpace"]["issueWeights"][issue] * \
                       self.data["LinearAdditiveUtilitySpace"]["issueUtilities"][issue]["DiscreteValueSetUtilities"][
                           "valueUtilities"][str(value)])
