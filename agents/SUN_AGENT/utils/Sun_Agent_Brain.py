import json
import random
import time

from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd
import lightgbm as lgb
from agents.SUN_AGENT.utils.profile_parser import ProfileParser

from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.issuevalue.Bid import Bid


class AgentBrain:
    def __init__(self, profile_parser_agent: ProfileParser, profile_parser_oppo: ProfileParser, ):

        self.sorted_bids_agent_that_greater_than_065_df = None
        self.sorted_bids_agent_that_greater_than_065 = []

        self.reservationBid_utility = float(0)
        self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent = None
        self.sorted_bids_agent_df = None
        self.reservationBid: Bid = None
        self.sorted_bids_agent = None
        self.sorted_bids_agent_that_greater_than_goal_of_utility = []
        self.all_bid_list = None

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

        self.number_of_bid_greater_than95 = 0
        self.percentage_of_greater_than95 = 0

        self.number_of_bid_greater_than85 = 0
        self.percentage_of_greater_than85 = 0

        self.goal_of_utility = 0.80
        self.number_of_goal_of_utility = None

    @staticmethod
    def get_goal_of_negoation_utility(x):
        return (float(-57.57067183) * float(x) * float(x) + float(x) * float(7.50261378) + float(1.59499339)) / float(2)

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
        if progress_time < 0.81:
            val = (float(0.99) - (float(0.14) * (float(progress_time))))
            """Y tarafına öyle bir değişken atamalıyım ki adamın utilitisi olmalı (kendi utilitime göre olsa daha mantıklı olabilir gibi şimdilik)"""
            new = pd.DataFrame([val])
            # Testing
            # new = pd.DataFrame([self.profile_parser_oppo.getUtility(bid)])

            self.Y = pd.concat([self.Y, new])

    def fill_domain_and_profile(self, domain, profile, profile_parser_opponent):
        self.domain = domain
        self.profile = profile
        self.reservationBid = self.profile.getReservationBid()
        if self.reservationBid is not None:
            self.reservationBid_utility = self.profile.getUtility(self.reservationBid)
        self.profile_parser_agent = profile

        self.profile_parser_oppo = profile_parser_opponent
        self.issue_name_list = self.domain.getIssues()
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.temEnumDict = self.enumerate_enum_dict()
        self.all_bid_list = AllBidsList(domain)

        self.sorted_bids_agent = sorted(self.all_bid_list,
                                        key=lambda x: self.profile.getUtility(x),
                                        reverse=True)
        self.calculate_percantage_and_number()
        self.add_agent_first_n_bid_to_machine_learning_with_low_utility(self.sorted_bids_agent)
        self.set_profile_test_data()

    def calculate_percantage_and_number(self):
        numb_95 = 0
        numb_85 = 0
        for i in self.sorted_bids_agent:
            if self.profile.getUtility(i) > float(0.95):
                numb_95 = numb_95 + 1
            if self.profile.getUtility(i) > float(0.85):
                numb_85 = numb_85 + 1
            else:
                break
        self.number_of_bid_greater_than95 = numb_95
        self.number_of_bid_greater_than85 = numb_85

        self.percentage_of_greater_than95 = float(self.number_of_bid_greater_than95) / float(
            len(self.sorted_bids_agent))
        self.percentage_of_greater_than85 = float(self.number_of_bid_greater_than85) / float(
            len(self.sorted_bids_agent))

        self.goal_of_utility = self.get_goal_of_negoation_utility(self.percentage_of_greater_than85) + float(0.01)
        print("goal " + str(self.goal_of_utility))
        numb_goal_util = 0
        self.sorted_bids_agent_df = pd.DataFrame()
        self.sorted_bids_agent_that_greater_than_065_df = pd.DataFrame()
        for i in self.sorted_bids_agent:
            utility = float(self.profile.getUtility(i))
            if utility > float(self.goal_of_utility):
                numb_goal_util = numb_goal_util + 1
            if utility > (float(self.goal_of_utility) - float(0.1)):
                self.sorted_bids_agent_that_greater_than_goal_of_utility.append(i)
                df_temp = pd.DataFrame(self.get_bid_value_array_for_data_frame_usage(i))
                df_temp = self.enumerate(df_temp)
                self.sorted_bids_agent_df = pd.concat([self.sorted_bids_agent_df, df_temp])
            if utility > 0.65:
                self.sorted_bids_agent_that_greater_than_065.append(i)
                df_temp = pd.DataFrame(self.get_bid_value_array_for_data_frame_usage(i))
                df_temp = self.enumerate(df_temp)
                self.sorted_bids_agent_that_greater_than_065_df = pd.concat([self.sorted_bids_agent_df, df_temp])
            else:
                break
        self.number_of_goal_of_utility = numb_goal_util

    def evaluate_opponent_utility_for_all_my_important_bid(self, progress_time):
        self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent = []
        util_of_opponent = self.lgb_model.predict(self.sorted_bids_agent_that_greater_than_065_df)

        for index, i in enumerate(self.sorted_bids_agent_that_greater_than_goal_of_utility):
            util = float(self.profile.getUtility(i))
            if float(self.reservationBid_utility) < util \
                    and (((float(0.95) - (
                    (float(0.95) - (self.goal_of_utility - float(0.1))) * float(progress_time))) < util)
                         and float(0.40) < util_of_opponent[index] < util - float(0.05)):
                self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent.append(i)

        if len(self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent) == 0 and 0.60 < progress_time < 0.72:
            for index, i in enumerate(self.sorted_bids_agent_that_greater_than_065):
                util = float(self.profile.getUtility(i))
                if float(self.reservationBid_utility) < util \
                        and util_of_opponent[index] < util - float(0.1):
                    self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent.append(i)

        print("leng" + str(len(self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent)))

    def evaluate_data_according_to_lig_gbm(self, progress_time):
        length = len(self.offers_unique)
        if length >= 1 and (length % 2) == 0:
            self.train_machine_learning_model()
            self.evaluate_opponent_utility_for_all_my_important_bid(progress_time)
            return self.test_machine_learning_model()

    def test_machine_learning_model(self):
        y_pred = self.lgb_model.predict(self.x_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        print("Mae" + str(mae) + "len" + str(len(self.offers_unique)))

        self.average_mse.append(float(mae))
        return float(mae)

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
                'learning_rate': 0.01,
                'force_row_wise': True,
                'feature_fraction': 1,
                'max_depth': 3,
                'num_leaves': 4,
                'boosting': 'gbdt',
                'min_data': 1,
                'verbose': -1
            }
        self.lgb_model = lgb.train(self.param, train_data)

    def call_model_lgb(self, bid):
        if self.lgb_model:
            prediction = self.lgb_model.predict(self._bid_for_model_prediction_to_df(bid))
            return prediction[0]
        else:
            return 1

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
        if float(self.profile_parser_oppo.getUtility(temp_bid)) > float(0.6):
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
        util = float(float(0.2) + (float(ratio) * float(0.35)))
        """Y tarafına öyle bir değişken atamalıyım ki adamın utilitisi olmalı (kendi utilitime göre olsa daha mantıklı olabilir gibi şimdilik)"""
        new = pd.DataFrame([util])

        self.Y = pd.concat([self.Y, new])

    def add_agent_first_n_bid_to_machine_learning_with_low_utility(self, sorted_bids_agent):

        """bid_number = 4
                if len(sorted_bids_agent) > 10000:
                    bid_number = 30
                elif len(sorted_bids_agent) > 5000:
                    bid_number = 25
                elif len(sorted_bids_agent) > 4000:
                    bid_number = 15
                elif len(sorted_bids_agent) > 2000:
                    bid_number = 7
                elif len(sorted_bids_agent) > 1000:
                    bid_number = 5"""
        if self.number_of_goal_of_utility > 150:
            bid_number = 40
            print("considered bid number " + str(bid_number) + "goal " + str(self.number_of_goal_of_utility))

        elif self.number_of_goal_of_utility > 100:
            bid_number = int(float(self.number_of_goal_of_utility) / float(3.4))
            print("considered bid number " + str(bid_number) + "goal " + str(self.number_of_goal_of_utility))

        elif self.number_of_goal_of_utility > 80:
            bid_number = int(float(self.number_of_goal_of_utility) / float(3.1))
            print("considered bid number " + str(bid_number) + "goal " + str(self.number_of_goal_of_utility))

        elif self.number_of_goal_of_utility > 50:
            bid_number = int(float(self.number_of_goal_of_utility) / float(3))
            print("considered bid number " + str(bid_number) + "goal " + str(self.number_of_goal_of_utility))

        elif self.number_of_goal_of_utility > 30:
            bid_number = 9
            print("considered bid number " + str(bid_number) + "goal " + str(self.number_of_goal_of_utility))

        elif self.number_of_goal_of_utility > 18:
            bid_number = 7
            print("considered bid number " + str(bid_number) + "goal " + str(self.number_of_goal_of_utility))
        elif 16 > self.number_of_goal_of_utility > 8:
            bid_number = int(float(self.number_of_goal_of_utility) / float(2))
            print("considered bid number " + str(bid_number) + "goal " + str(self.number_of_goal_of_utility))
        else:
            print("else")
            bid_number = 4
        for i in range(0, bid_number + 1):
            bid = sorted_bids_agent[i]
            self.util_add_agent_first_n_bid_to_machine_learning_with_low_utility(bid,
                                                                                 float(float(i) / float(bid_number)))

    def is_acceptable(self, bid: Bid, progress):
        if self.profile.getUtility(bid) > 0.94:
            return True
        elif 0.94 > progress > 0.80 and self.profile.getUtility(bid) > self.goal_of_utility and self.profile.getUtility(
                bid) - float(0.15) > float(self.call_model_lgb(bid)):
            return True
        elif 0.97 > progress > 0.94 and self.profile.getUtility(bid) > self.goal_of_utility - float(0.15):
            return True
        return False

    def find_bid(self, progress_time):
        progress_time = float(progress_time)
        if 0 < progress_time < 0.9 and self.lgb_model is not None and len(
                self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent) >= 1:
            index = random.randint(0,
                                   len(self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent) - 1)
            if float(self.reservationBid_utility) < float(
                    self.profile.getUtility(self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent[index])):
                return self.eva_util_val_acc_to_lgb_m_with_max_bids_for_agent[index]
        elif progress_time < 0.4:
            if self.number_of_bid_greater_than95 >= 4:
                index = random.randint(3, self.number_of_bid_greater_than95)
                if float(self.reservationBid_utility) < float(self.profile.getUtility(self.sorted_bids_agent[index])):
                    return self.sorted_bids_agent[index]
            if self.number_of_bid_greater_than95 >= 1:
                index = random.randint(1, self.number_of_bid_greater_than95)
                if float(self.reservationBid_utility) < float(self.profile.getUtility(self.sorted_bids_agent[index])):
                    return self.sorted_bids_agent[index]

            elif self.number_of_bid_greater_than85 >= 1:
                index = random.randint(1, self.number_of_bid_greater_than85)
                if float(self.reservationBid_utility) < float(self.profile.getUtility(self.sorted_bids_agent[index])):
                    return self.sorted_bids_agent[index]

        elif progress_time < 0.85:
            if self.number_of_bid_greater_than95 > 1 and self.number_of_bid_greater_than85 > 2:
                index = random.randint(self.number_of_bid_greater_than95, self.number_of_bid_greater_than85)
                if float(self.reservationBid_utility) < float(self.profile.getUtility(self.sorted_bids_agent[index])):
                    return self.sorted_bids_agent[index]

            elif self.number_of_bid_greater_than85 >= 1:
                index = random.randint(1, self.number_of_bid_greater_than85)
                if float(self.reservationBid_utility) < float(self.profile.getUtility(self.sorted_bids_agent[index])):
                    return self.sorted_bids_agent[index]

        elif progress_time <= 0.975:
            if self.number_of_goal_of_utility > self.number_of_bid_greater_than85:
                index = random.randint(self.number_of_bid_greater_than85, self.number_of_goal_of_utility)
                if float(self.reservationBid_utility) < float(self.profile.getUtility(self.sorted_bids_agent[index])):
                    return self.sorted_bids_agent[index]
            elif self.number_of_goal_of_utility > self.number_of_bid_greater_than95:
                index = random.randint(self.number_of_bid_greater_than95, self.number_of_goal_of_utility)
                if float(self.reservationBid_utility) < float(self.profile.getUtility(self.sorted_bids_agent[index])):
                    return self.sorted_bids_agent[index]
            elif self.number_of_goal_of_utility > 1:
                index = random.randint(1, self.number_of_goal_of_utility)
                if float(self.reservationBid_utility) < float(self.profile.getUtility(self.sorted_bids_agent[index])):
                    return self.sorted_bids_agent[index]
        elif progress_time <= 0.998:
            sorted_opponent_bid_acc_to_agent = sorted(self.offers_unique, key=lambda x: self.profile.getUtility(x),
                                                      reverse=True)
            bid = sorted_opponent_bid_acc_to_agent[0]
            util_of_bid = self.profile.getUtility(bid)
            if float(self.reservationBid_utility) < float(util_of_bid) and float(util_of_bid) >= float(
                    self.goal_of_utility) - float(0.03) and self.call_model_lgb(bid) < util_of_bid:
                return bid
        return self.sorted_bids_agent[3]


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
