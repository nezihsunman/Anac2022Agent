from decimal import Decimal

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import numpy as np
from copy import copy
import pandas as pd

from agents.SUN_AGENT.utils.profile_parser import ProfileParser

from geniusweb.bidspace.AllBidsList import AllBidsList


class GradientBoostingRegressorModel:
    def __init__(self, profile_parser: ProfileParser):
        self.clf = GradientBoostingRegressor(random_state=10)
        """np.random.random((10000, 9))"""
        self.X = None
        self.Y = None
        self.domain = None
        self.profile = None
        self.issue_name_list = None
        self.profile_parser = profile_parser

    def __add__(self, other):
        reg = GradientBoostingRegressor(random_state=0)
        reg.fit(self.X, self.Y)

    def add_opponent_offer_to_x(self, bid):
        # Gönderilen bid benim için 0.6 ve 0.9 arasında ise oponent modele ekle
        if Decimal(0.3) < self.profile.getUtility(bid) < Decimal(1):
            bid_value_array = self.getBidValueArrayForDataFrameUsage(bid)
            df = pd.DataFrame(bid_value_array)
            self.X = pd.concat([self.X, df])

            """Y tarafına öyle bir değişken atamalıyım ki adamın utilitisi olmalı (kendi utilitime göre olsa daha mantıklı olabilir gibi şimdilik)"""
            new = pd.DataFrame([(Decimal(10) / Decimal(10)) - self.profile.getUtility(bid)])

            self.Y = pd.concat([self.Y, new])

    def getBidValueArrayForDataFrameUsage(self, bid):
        bid_value_array = {}
        for issue in self.issue_name_list:
            bid_value_array[issue] = [bid.getValue(issue)]
        return bid_value_array

    def add_domain_and_profile(self, domain, profile):
        self.domain = domain
        self.profile = profile
        self.issue_name_list = self.domain.getIssues()
        self.X = pd.DataFrame(None)
        self.Y = pd.DataFrame(None)
        self.clf = GradientBoostingRegressor(random_state=10)

    def evaluate_model_and_compare(self):
        # Enumaration

        issue_enums = dict((y, x) for x, y in enumerate(set(self.domain.getIssues())))
        enums_2 = dict((y, x) for x, y in enumerate(set(self.domain.getIssuesValues())))

        enums = dict((y, x) for x, y in enumerate(zip(self.domain.getIssues(), self.domain.getIssuesValues())))
        issue_enums_dict = {}
        for issue in self.domain.getIssues():
            temp_enums = dict((y, x) for x, y in enumerate(set(self.domain.getIssuesValues()[issue])))
            issue_enums_dict[issue] = temp_enums
            self.X[issue] = self.X[issue].map(temp_enums)

        self.clf.fit(self.X, self.Y)

        y_test = pd.DataFrame()
        x_test = pd.DataFrame()
        total_utility = []
        all_bids = AllBidsList(self.domain)
        # Yarışmada kaldırılcak
        for i in range(all_bids.size() - 1):
            temp_bid = all_bids.get(i)
            utility_of_bid_age = self.profile.getUtility(temp_bid)
            if Decimal(0.3) < utility_of_bid_age < Decimal(0.9):
                df_temp = pd.DataFrame(self.getBidValueArrayForDataFrameUsage(temp_bid))
                x_test = pd.concat([x_test, df_temp])
                utility_of_bid = self.profile_parser.getUtility(temp_bid)
                total_utility.append(utility_of_bid_age + utility_of_bid)
                new = pd.DataFrame([utility_of_bid])
                y_test = pd.concat([y_test, new])

        mean = np.mean(total_utility)
        for issue in self.domain.getIssues():
            x_test[issue] = x_test[issue].map(issue_enums_dict[issue])

        self.clf.predict(x_test)

        score = self.clf.score(x_test, y_test)
        print(score)

    def fit(self):
        self.clf.fit(self.X, self.Y)

    def optimize(self):
        x0 = len([0.5] * self.domain.getIssuesValues())
        initial_simplex = self._get_simplex(x0, 0.1)
        result = minimize(fun=self._call_model,
                          x0=np.array(x0),
                          method='Nelder-Mead',
                          options={'xatol': 0.1,
                                   'initial_simplex': np.array(initial_simplex)})
        return result

    def _call_model(self, x):
        prediction = self.clf.predict([x])
        return prediction[0]

    def _get_simplex(self, x0, step):
        simplex = []
        for i in range(len(x0)):
            point = copy(x0)
            point[i] -= step
            simplex.append(point)

        point2 = copy(x0)
        point2[-1] += step
        simplex.append(point2)
        return simplex

    """
            reg.predict(X_test[1:2])
    
            reg.score(X_test, y_test)
    """
