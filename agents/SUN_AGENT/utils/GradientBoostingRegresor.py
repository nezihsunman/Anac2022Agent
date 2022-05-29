import json
import statistics
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import pandas as pd
import lightgbm as lgb
from bisect import bisect

from agents.SUN_AGENT.utils.profile_parser import ProfileParser

from geniusweb.bidspace.AllBidsList import AllBidsList


class GradientBoostingRegressorModel:
    def __init__(self, profile_parser_agent: ProfileParser, profile_parser_oppo: ProfileParser, ):
        self.num_round = 100
        self.param = None
        # dont Forget verbose in tornamet
        num_round = 100
        self.y_test = None
        self.x_test = None
        self.lgb_model = None
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.domain = None
        self.profile = None
        self.issue_name_list = None
        self.profile_parser_agent = profile_parser_agent
        self.profile_parser_oppo = profile_parser_oppo
        self.temEnumDict = None

        self.average_mse = []

    def add_opponent_offer_to_x(self, bid, progress_time):
        # Gönderilen bid benim için 0.6 ve 0.9 arasında ise oponent modele ekle
        # if Decimal(0.3) < self.profile.getUtility(bid) < Decimal(1):
        bid_value_array = self.get_bid_value_array_for_data_frame_usage(bid)
        df = pd.DataFrame(bid_value_array)
        df = self.enumerate(df)
        self.X = pd.concat([self.X, df])

        """Y tarafına öyle bir değişken atamalıyım ki adamın utilitisi olmalı (kendi utilitime göre olsa daha mantıklı olabilir gibi şimdilik)"""
        new = pd.DataFrame([(float(1) - (float(0.2) * (float(progress_time))))])
        new = pd.DataFrame([self.profile_parser_oppo.getUtility(bid)])

        self.Y = pd.concat([self.Y, new])

    def get_bid_value_array_for_data_frame_usage(self, bid):
        bid_value_array = {}
        for issue in self.issue_name_list:
            bid_value_array[issue] = [bid.getValue(issue)]
        return bid_value_array

    def add_domain_and_profile(self, domain, profile, profile_parser_opponent):
        self.domain = domain
        self.profile = profile
        self.profile_parser_agent = profile

        self.profile_parser_oppo = profile_parser_opponent
        self.issue_name_list = self.domain.getIssues()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.temEnumDict = self.enumerate_enum_dict()
        self.set_profile_test_data()

    def evaluate_data_according_to_lig_gbm(self):
        issueList = []
        for issue in self.domain.getIssues():
            issueList.append(issue)
        for col in issueList:
            self.X[col] = self.X[col].astype('int')
        self.Y = self.Y.astype('float')
        train_data = lgb.Dataset(self.X, label=self.Y, feature_name=issueList)

        if self.param is None:
            self.param = {
                'objective': 'regression',
                'learning_rate': 0.05,
                'force_row_wise': True,
                'feature_fraction': 1,
                'max_depth': len(self.issue_name_list),
                'num_leaves': int((2 ** len(self.issue_name_list) / 4)),
                'boosting': 'gbdt',
                'min_data': 1,
                'verbose': -1
            }

        self.lgb_model = lgb.train(self.param, train_data, self.num_round)

        y_pred = self.lgb_model.predict(self.x_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        self.average_mse.append(float(mae))
        return mae

    """def _call_model_gradient_b_r(self, x):
        prediction = self.gradientBoostingRegressor.predict(self._bid_to_enumerated_df(x))
        return prediction[0]"""

    def _call_model_lgb(self, bid):
        if self.lgb_model:
            prediction = self.lgb_model.predict(self._bid_to_enumerated_df(bid))
            return prediction[0]
        else:
            return 0

    def _bid_to_enumerated_df(self, bid):
        df_temp = pd.DataFrame(self.get_bid_value_array_for_data_frame_usage(bid))
        df_temp = self.enumerate(df_temp)
        return df_temp

    def enumerate_enum_dict(self):
        issue_enums_dict = {}
        for issue in self.domain.getIssues():
            temp_enums = dict((y, x) for x, y in enumerate(set(self.domain.getIssuesValues()[issue])))
            issue_enums_dict[issue] = temp_enums
        return issue_enums_dict

    def enumerate(self, df):
        for issue in self.domain.getIssues():
            df[issue] = df[issue].map(self.temEnumDict[issue])
        return df

    def get_average_of_mae(self):
        if len(self.average_mse) == 0:
            return float(-1)
        return np.mean(self.average_mse)

    def set_profile_test_data(self):
        self.x_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        all_bids = AllBidsList(self.domain)
        for i in range(0, all_bids.size() - 1, 1):
            temp_bid = all_bids.get(i)
            self.calculate_opponent_profile_for_test(temp_bid)

    def calculate_opponent_profile_for_test(self, temp_bid):
        if float(self.profile_parser_oppo.getUtility(temp_bid)) > float(0.72):
            df_temp = pd.DataFrame(self.get_bid_value_array_for_data_frame_usage(temp_bid))
            utility_of_bid = self.profile_parser_oppo.getUtility(temp_bid)

            new = pd.DataFrame([utility_of_bid])
            self.y_test = pd.concat([self.y_test, new])

            df_temp = self.enumerate(df_temp)
            self.x_test = pd.concat([self.x_test, df_temp])

    def model_feature_importance(self):
        df = pd.DataFrame({'Value': self.lgb_model.feature_importance(), 'Feature': self.X.columns})
        df = pd.DataFrame({'Value': self.lgb_model.feature_importance(), 'Feature': self.X.columns})
        result = df.to_json(orient="split")
        parsed = json.loads(result)
        return parsed

    def util_add_agent_first_n_bid_to_machine_learning_with_low_utility(self, bid, ratio):
        bid_value_array = self.get_bid_value_array_for_data_frame_usage(bid)
        df = pd.DataFrame(bid_value_array)
        df = self.enumerate(df)
        self.X = pd.concat([self.X, df])
        util = float(float(0.07) + (float(ratio) * float(0.35)))
        """Y tarafına öyle bir değişken atamalıyım ki adamın utilitisi olmalı (kendi utilitime göre olsa daha mantıklı olabilir gibi şimdilik)"""
        new = pd.DataFrame([util])

        self.Y = pd.concat([self.Y, new])

    def add_agent_first_n_bid_to_machine_learning_with_low_utility(self, sorted_bids_agent):
        bid_number = 4
        if len(sorted_bids_agent) > 10000:
            bid_number = 50
        elif len(sorted_bids_agent) > 5000:
            bid_number = 35
        elif len(sorted_bids_agent) > 4000:
            bid_number = 25
        elif len(sorted_bids_agent) > 2000:
            bid_number = 15
        elif len(sorted_bids_agent) > 1000:
            bid_number = 10
        for i in range(1, bid_number + 1):
            bid = sorted_bids_agent[i]
            self.util_add_agent_first_n_bid_to_machine_learning_with_low_utility(bid,
                                                                                 float(float(i) / float(bid_number)))


"""def evaluate_model_and_compare(self):
    # Enumaration
    if not (self.X.isnull().values.any() or self.Y.isnull().values.any()):
        self.gradientBoostingRegressor.fit(self.X, self.Y)
    else:
        print("DF contains zero values")
    y_test = pd.DataFrame()
    x_test = pd.DataFrame()
    total_utility = []
    all_bids = AllBidsList(self.domain)
    # Yarışmada kaldırılcak
    for i in range(all_bids.size() - 1):
        temp_bid = all_bids.get(i)
        utility_of_bid_age = self.profile.getUtility(temp_bid)
        # if Decimal(0.2) < utility_of_bid_age < Decimal(0.99):
        df_temp = pd.DataFrame(self.get_bid_value_array_for_data_frame_usage(temp_bid))
        x_test = pd.concat([x_test, df_temp])
        utility_of_bid = self.profile_parser.getUtility(temp_bid)
        total_utility.append(utility_of_bid_age + utility_of_bid)
        new = pd.DataFrame([utility_of_bid])
        y_test = pd.concat([y_test, new])

    mean = np.mean(total_utility)
    for issue in self.domain.getIssues():
        x_test[issue] = x_test[issue].map(self.temEnumDict[issue])
    if x_test.isnull().values.any():
        print("Xtest : " + x_test.keys())

    print(score)
    return score
"""
