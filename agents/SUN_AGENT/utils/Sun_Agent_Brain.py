import json
from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd
import lightgbm as lgb
from agents.SUN_AGENT.utils.profile_parser import ProfileParser

from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.issuevalue.Bid import Bid


class AgentBrain:
    def __init__(self, profile_parser_agent: ProfileParser, profile_parser_oppo: ProfileParser, ):
        self.reservationBid: Bid = None
        self.sorted_bids_agent = None
        self.all_bid_list = None

        self.num_round = 100
        self.param = None

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

        self.offers = []
        self.offers_unique = []

    def keep_opponent_offer_in_a_list(self, bid: Bid):
        # keep track of all bids received
        self.offers.append(bid)

        if bid not in self.offers_unique:
            self.offers_unique.append(bid)

    def add_opponent_offer_to_self_x_and_self_y(self, bid, progress_time):
        bid_value_array = self.get_bid_value_array_for_data_frame_usage(bid)
        df = pd.DataFrame(bid_value_array)
        df = self.enumerate(df)
        self.X = pd.concat([self.X, df])

        """Y tarafına öyle bir değişken atamalıyım ki adamın utilitisi olmalı (kendi utilitime göre olsa daha mantıklı olabilir gibi şimdilik)"""
        new = pd.DataFrame([(float(0.95) - (float(0.2) * (float(progress_time))))])
        # Testing
        # new = pd.DataFrame([self.profile_parser_oppo.getUtility(bid)])

        self.Y = pd.concat([self.Y, new])

    def fill_domain_and_profile(self, domain, profile, profile_parser_opponent):
        self.domain = domain
        self.profile = profile
        self.reservationBid = self.profile.getReservationBid()
        self.profile_parser_agent = profile

        self.profile_parser_oppo = profile_parser_opponent
        self.issue_name_list = self.domain.getIssues()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.temEnumDict = self.enumerate_enum_dict()
        self.set_profile_test_data()
        self.all_bid_list = AllBidsList(domain)

        self.sorted_bids_agent = sorted(self.all_bid_list,
                                        key=lambda x: self.profile.getUtility(x),
                                        reverse=True)

    def evaluate_data_according_to_lig_gbm(self):
        self.train_machine_learning_model()
        return self.test_machine_learning_model()

    def test_machine_learning_model(self):
        y_pred = self.lgb_model.predict(self.x_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        self.average_mse.append(float(mae))
        return mae

    def train_machine_learning_model(self):
        issueList = []
        for issue in self.domain.getIssues():
            issueList.append(issue)
        for col in issueList:
            self.X[col] = self.X[col].astype('int')
        self.Y = self.Y.astype('float')
        train_data = lgb.Dataset(self.X, label=self.Y, feature_name=issueList)
        if self.param is None:
            objective = ['cross_entropy', 'lambdarank', 'regression', 'huber', 'mape']
            self.param = {
                'objective': 'cross_entropy',
                'learning_rate': 0.05,
                'force_row_wise': True,
                'feature_fraction': 1,
                'max_depth': 2,
                'num_leaves': 4,
                'boosting': 'gbdt',
                'min_data': 1,
                'verbose': -1
            }
        self.lgb_model = lgb.train(self.param, train_data, self.num_round)

    def _call_model_lgb(self, bid):
        if self.lgb_model:
            prediction = self.lgb_model.predict(self._bid_for_model_prediction_to_df(bid))
            return prediction[0]
        else:
            return 0

    def get_bid_value_array_for_data_frame_usage(self, bid):
        bid_value_array = {}
        for issue in self.issue_name_list:
            bid_value_array[issue] = [bid.getValue(issue)]
        return bid_value_array

    def _bid_for_model_prediction_to_df(self, bid):
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
        """self.train_machine_learning_model()
        real_utility_of_opponent = self.profile_parser_oppo.getUtility(bid)
        estimated_utility_before_adding_data = self.lgb_model.predict(self._bid_for_model_prediction_to_df(bid))"""
        bid_value_array = self.get_bid_value_array_for_data_frame_usage(bid)
        df = pd.DataFrame(bid_value_array)
        df = self.enumerate(df)
        self.X = pd.concat([self.X, df])
        util = float(float(0.3) + (float(ratio) * float(0.35)))
        """Y tarafına öyle bir değişken atamalıyım ki adamın utilitisi olmalı (kendi utilitime göre olsa daha mantıklı olabilir gibi şimdilik)"""
        new = pd.DataFrame([util])

        self.Y = pd.concat([self.Y, new])

    def add_agent_first_n_bid_to_machine_learning_with_low_utility(self, sorted_bids_agent):
        bid_number = 4
        if len(sorted_bids_agent) > 10000:
            bid_number = 30
        elif len(sorted_bids_agent) > 5000:
            bid_number = 25
        elif len(sorted_bids_agent) > 4000:
            bid_number = 15
        elif len(sorted_bids_agent) > 2000:
            bid_number = 7
        elif len(sorted_bids_agent) > 1000:
            bid_number = 5
        for i in range(0, bid_number + 1):
            bid = sorted_bids_agent[i]
            self.util_add_agent_first_n_bid_to_machine_learning_with_low_utility(bid,
                                                                                 float(float(i) / float(bid_number)))

    def is_acceptable(self, bid: Bid, progress):
        if self.profile.getUtility(bid) > 0.95:
            return True

    def find_bid(self, progress_time):
        return self.sorted_bids_agent[0]


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