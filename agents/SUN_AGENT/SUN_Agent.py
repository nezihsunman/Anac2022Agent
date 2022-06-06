import json
import logging
from random import randint
from time import time
from typing import cast, Dict, List, Union, AnyStr, Set

import numpy as np
import pandas as pd
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.ValueSet import ValueSet
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft.utilities.immutablelist.ImmutableList import ImmutableList
from tudelft.utilities.immutablelist.Outer import Outer
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from agents.SUN_AGENT.utils.Sun_Agent_Brain import AgentBrain
from agents.SUN_AGENT.utils.profile_parser import ProfileParser
from timeit import default_timer as timer


class SunAgent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.this_session_is_first_match_for_this_opponent = True
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.sorted_bids = None

        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.opponent_id: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None

        self.profile_opponent_parser = ProfileParser()
        self.profile_parser_opponent = ProfileParser()

        self.agent_brain = AgentBrain(self.profile_opponent_parser,
                                      self.profile_parser_opponent)

        self.storage_data = {}
        self.isFirstRound = True
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")
            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            all_bids = AllBidsList(self.domain)
            if not self.sorted_bids:
                self.sorted_bids = sorted(all_bids, key=lambda x: self.profile.getUtility(x),
                                          reverse=True)
            """Agent Model"""
            path = ""
            if 'profileA' == self.profile.getName():
                path = "domains/" + self.domain.getName() + "/profileB.json"
            else:
                path = "domains/" + self.domain.getName() + "/profileA.json"

            self.profile_opponent_parser.parse(path)

            self.agent_brain.fill_domain_and_profile(self.domain, self.profile,
                                                     self.profile_opponent_parser)

            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.opponent_id = str(actor).rsplit("_", 1)[0]

                if self.isFirstRound:
                    self.load_data()
                    self.isFirstRound = False
                # process action done by opponent
                self.opponent_action(action)

        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            bid = cast(Offer, action).getBid()

            if bid not in self.agent_brain.offers_unique:
                # bid util
                if self.profile.getUtility(bid) > self.agent_brain.profile_parser_oppo.getUtility(bid):
                    print("I can win")
                    print("my uti" + str(self.agent_brain.profile.getUtility(bid)))
                    print("my oppo" + str(self.agent_brain.profile_parser_oppo.getUtility(bid)))
                    print("my oppo" + str(self.agent_brain.call_model_lgb(bid)))

                self.agent_brain.add_opponent_offer_to_self_x_and_self_y(bid,
                                                                         self.progress.get(time() * 1000))
                if len(self.agent_brain.offers_unique) <= 16 and self.progress.get(time() * 1000) < 0.81:
                    self.agent_brain.evaluate_data_according_to_lig_gbm(self.progress.get(time() * 1000))
            self.agent_brain.keep_opponent_offer_in_a_list(bid)

            # update opponent model with bid

            self.profile_opponent_parser.getUtility(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            progress_time = self.progress.get(time() * 1000)
            bid = self.agent_brain.find_bid(progress_time)
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        if 'offerNumberUnique' in self.storage_data.keys():
            self.storage_data['offerNumberUnique'].append(len(self.agent_brain.offers_unique))
        else:
            self.storage_data['offerNumberUnique'] = [len(self.agent_brain.offers_unique)]
        if 'domainName' in self.storage_data.keys():
            self.storage_data['domainName'].append(self.domain.getName())
        else:
            self.storage_data['domainName'] = [self.domain.getName()]

        print("uniq" + str(len(self.agent_brain.offers_unique)))
        mae = self.agent_brain.get_average_of_mae()
        print("Average mae" + str(mae))
        self.store_rmse_in_local_storage(mae)

        print("I save data to storage")
        with open(f"{self.storage_dir}/{self.opponent_id}data.md", "w") as f:
            f.write(json.dumps(self.storage_data))

    def load_data(self):
        if self.opponent_id is not None and self.storage_dir is not None:
            try:
                with open(self.storage_dir + "/" + self.opponent_id + "data.md") as file:
                    self.storage_data = json.load(file)
                    print("I load data from storage")
                    self.this_session_is_first_match_for_this_opponent = False
            except Exception:
                pass

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        return self.agent_brain.is_acceptable(bid, progress)

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        """time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.agent_brain is not None:
            opponent_utility = self.agent_brain.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score"""

        return our_utility

    def store_rmse_in_local_storage(self, rmse):
        if 'mae' in self.storage_data.keys():
            self.storage_data['mae'].append(rmse)
        else:
            self.storage_data['mae'] = [rmse]


"""if __name__ == "__main__":
    agent = SunAgent()
    opponent = SunAgent()

    domain = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"
    domain_path_1 = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"

    profileJsonOfOpponent = "/profileA.json"
    profileJsonOfAgent = "/profileB.json"
    json_path = ".json"

    result = []
    for i in range(0, 50):
        stringNumber = str(i).zfill(2)
        print(stringNumber)
        domain_opponent = domain + stringNumber + profileJsonOfOpponent
        domain_agent = domain + stringNumber + profileJsonOfAgent

        profile_parser_agent = ProfileParser()
        profile_parser_agent.parse(domain_agent)
        profile_parser_opponent = ProfileParser()
        profile_parser_opponent.parse(domain_opponent)

        domain_path = domain_path_1 + stringNumber + "/domain" + stringNumber + json_path

        with open(domain_path) as file:
            domain_data = json.load(file)
        name = "domain_name"
        issue_values: Dict[str, ValueSet] = {}
        for issue_dict in domain_data['issuesValues'].keys():
            mp: List[ImmutableList[Value]] = []
            for value in domain_data['issuesValues'][issue_dict]['values']:
                mp.append(cast(Value, value))
            issue_values[issue_dict] = cast(ImmutableList[Value], mp)
            # issue_values[issue_dict] = cast(Value,issue_values[issue_dict]['values'])

        domain_class = Domain(name, issue_values)
        """
"""issues: List[Set[str]] = list(domain_class.getIssues())
values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issues]
all_bids: Outer = Outer[Value](values)"""
"""
        for issue_dict in issue_values:
            values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issue_values]
            issue_values[issue_dict]: List[ImmutableList[Value]] = values
        all_bids_list = AllBidsList(domain_class)

        sorted_bids_agent = sorted(all_bids_list, key=lambda x: profile_parser_opponent.getUtility_for_testing(x),
                                   reverse=True)
        sorted_bids_opponent = sorted(all_bids_list, key=lambda x: profile_parser_agent.getUtility_for_testing(x),
                                      reverse=True)

        opponent_reg = GradientBoostingRegressorModel(profile_parser_opponent)
        agent_reg = GradientBoostingRegressorModel(profile_parser_agent)

        opponent_reg.add_domain_and_profile(domain_class, profile_parser_opponent)
        agent_reg.add_domain_and_profile(domain_class, profile_parser_agent)

        bid_list_agent = []
        bid_number_that_random = 10
        for i in range(1, bid_number_that_random):
            bid_index = randint(0, int(len(sorted_bids_agent) * 0.001))
            bid = sorted_bids_agent[bid_index]
            bid_list_agent.append(bid)
            agent_reg.add_opponent_offer_to_x(bid, 0.1)

        bid_list_opponent = []

        for i in range(1, bid_number_that_random):
            bid_index = randint(0, int(len(sorted_bids_opponent) * 0.001))
            bid = sorted_bids_opponent[bid_index]
            bid_list_opponent.append(bid)
            opponent_reg.add_opponent_offer_to_x(bid, 0.1)

        for i in range(20):
            param = {
                'objective': 'regression',
                'learning_rate': 0.05,
                'force_row_wise': True,
                'feature_fraction': 1,
                'max_depth': i,
                'num_leaves': (2 ** i) / 2,
                'boosting': 'gbdt',
                'min_data': 1,
                'verbose': -1
            }

            agent_reg.param = param
            opponent_reg.param = param

            mae_agent = agent_reg.evaluate_data_according_to_lig_gbm()
            mae_opponent = agent_reg.evaluate_data_according_to_lig_gbm()
            dictionary = {'forEach': param, 'mae_agent': mae_agent, 'mae_opponent': mae_opponent
                , 'profile.agent': domain_agent,
                          'profile.oppo': domain_opponent,
                          'taken_random_bid_number_from_all_bid_list': bid_number_that_random}

            print(dictionary)

            result.append(dictionary)

    with open(f"data.md", "w") as f:
        f.write(json.dumps(result))
""""""
if __name__ == "__main__":
    agent = SunAgent()
    opponent = SunAgent()

    domain = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"
    domain_path_1 = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"

    profileJsonOfOpponent = "/profileA.json"
    profileJsonOfAgent = "/profileB.json"
    json_path = ".json"

    result = []
    param_result = []
    for i in range(10):
        for k in range(2, 50, 5):
            param = {
                'objective': 'regression',
                'learning_rate': 0.05,
                'force_row_wise': True,
                'feature_fraction': 1,
                'max_depth': i,
                'num_leaves': k,
                'boosting': 'gbdt',
                'min_data': 1,
                'verbose': -1
            }
            average_mae_agent = []
            average_mae_oppo = []

            for j in range(10, 50, 10):
                bid_number_that_random = j

                for i in range(0, 50):
                    stringNumber = str(i).zfill(2)
                    print(stringNumber)
                    domain_opponent = domain + stringNumber + profileJsonOfOpponent
                    domain_agent = domain + stringNumber + profileJsonOfAgent

                    profile_parser_agent = ProfileParser()
                    profile_parser_agent.parse(domain_agent)
                    profile_parser_opponent = ProfileParser()
                    profile_parser_opponent.parse(domain_opponent)

                    domain_path = domain_path_1 + stringNumber + "/domain" + stringNumber + json_path

                    with open(domain_path) as file:
                        domain_data = json.load(file)
                    name = "domain_name"
                    issue_values: Dict[str, ValueSet] = {}
                    for issue_dict in domain_data['issuesValues'].keys():
                        mp: List[ImmutableList[Value]] = []
                        for value in domain_data['issuesValues'][issue_dict]['values']:
                            mp.append(cast(Value, value))
                        issue_values[issue_dict] = cast(ImmutableList[Value], mp)
                        # issue_values[issue_dict] = cast(Value,issue_values[issue_dict]['values'])

                    domain_class = Domain(name, issue_values)
                    """"""issues: List[Set[str]] = list(domain_class.getIssues())
                    values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issues]
                    all_bids: Outer = Outer[Value](values)""""""
                    for issue_dict in issue_values:
                        values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issue_values]
                        issue_values[issue_dict]: List[ImmutableList[Value]] = values
                    all_bids_list = AllBidsList(domain_class)

                    sorted_bids_agent = sorted(all_bids_list,
                                               key=lambda x: profile_parser_agent.getUtility_for_testing(x),
                                               reverse=True)
                    sorted_bids_opponent = sorted(all_bids_list,
                                                  key=lambda x: profile_parser_opponent.getUtility_for_testing(x),
                                                  reverse=True)

                    opponent_reg = GradientBoostingRegressorModel(profile_parser_opponent)
                    agent_reg = GradientBoostingRegressorModel(profile_parser_agent)

                    opponent_reg.add_domain_and_profile(domain_class, profile_parser_opponent)
                    agent_reg.add_domain_and_profile(domain_class, profile_parser_agent)

                    agent_reg.param = param
                    opponent_reg.param = param

                    bid_list_agent = []
                    for i in range(1, bid_number_that_random):
                        if len(sorted_bids_agent) > 5000:
                            bid_index = randint(0, int(50))
                            bid = sorted_bids_agent[bid_index]
                            bid_list_agent.append(bid)
                            agent_reg.add_opponent_offer_to_x(bid, 0.1)
                        elif len(sorted_bids_agent) > 1000:
                            bid_index = randint(0, int(20))
                            bid = sorted_bids_agent[bid_index]
                            bid_list_agent.append(bid)
                            agent_reg.add_opponent_offer_to_x(bid, 0.1)
                        elif len(sorted_bids_agent) > 20:
                            bid_index = randint(0, int(10))
                            bid = sorted_bids_agent[bid_index]
                            bid_list_agent.append(bid)
                            agent_reg.add_opponent_offer_to_x(bid, 0.1)
                        else:
                            bid_index = randint(0, int(5))
                            bid = sorted_bids_agent[bid_index]
                            bid_list_agent.append(bid)
                            agent_reg.add_opponent_offer_to_x(bid, 0.1)
                    bid_list_opponent = []
                    for i in range(1, bid_number_that_random):
                        if len(sorted_bids_agent) > 5000:
                            bid_index = randint(0, int(50))
                            bid = sorted_bids_opponent[bid_index]
                            bid_list_agent.append(bid)
                            opponent_reg.add_opponent_offer_to_x(bid, 0.1)
                        elif len(sorted_bids_agent) > 1000:
                            bid_index = randint(0, int(20))
                            bid = sorted_bids_opponent[bid_index]
                            bid_list_agent.append(bid)
                            opponent_reg.add_opponent_offer_to_x(bid, 0.1)
                        elif len(sorted_bids_agent) > 20:
                            bid_index = randint(0, int(10))
                            bid = sorted_bids_opponent[bid_index]
                            bid_list_agent.append(bid)
                            opponent_reg.add_opponent_offer_to_x(bid, 0.1)
                        else:
                            bid_index = randint(0, int(1))
                            bid = sorted_bids_opponent[bid_index]
                            bid_list_agent.append(bid)
                            opponent_reg.add_opponent_offer_to_x(bid, 0.1)

                    agent_reg.param = param
                    opponent_reg.param = param

                    mae_agent = agent_reg.evaluate_data_according_to_lig_gbm()
                    mae_opponent = opponent_reg.evaluate_data_according_to_lig_gbm()

                    average_mae_agent.append(mae_agent)
                    average_mae_oppo.append(mae_opponent)
                    dictionary = {'forEach': param, 'mae_agent': mae_agent, 'mae_opponent': mae_opponent
                        , 'profile.agent': domain_agent,
                                  'profile.oppo': domain_opponent,
                                  'taken_random_bid_number_from_all_bid_list': bid_number_that_random}

                    print(dictionary)

                    result.append(dictionary)

                print("Average result for each parameter: ")
                print(param)
                avg_mea_agent_for_param = np.mean(average_mae_agent)
                avg_mea_oppo_for_param = np.mean(average_mae_oppo)
                dictionary2 = {'param': param, 'average_mae_agent': avg_mea_agent_for_param,
                               'average_mae_oppo': avg_mea_oppo_for_param}
                param_result.append(dictionary2)

    with open(f"result1.md", "w") as f:
        f.write(json.dumps(result))
    with open(f"param_result_1.md", "w") as f:
        f.write(json.dumps(param_result))
"""


def deep_analysis_for_machine_learning():
    agent = SunAgent()
    opponent = SunAgent()
    domain = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"
    domain_path_1 = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"
    profileJsonOfOpponent = "/profileA.json"
    profileJsonOfAgent = "/profileB.json"
    json_path = ".json"
    result = []
    param_result = []
    objective = ['cross_entropy', 'regression']
    for objective in objective:
        param = {
            'objective': objective,
            'learning_rate': 0.05,
            'force_row_wise': True,
            'feature_fraction': 1,
            'max_depth': 2,
            'num_leaves': 4,
            'boosting': 'gbdt',
            'min_data': 1,
            'verbose': -1
        }
        average_mae_agent = []
        average_mae_oppo = []

        for j in [4, 8, 12]:
            bid_number_that_random = j

            for k in range(0, 50):
                stringNumber = str(k).zfill(2)
                print(stringNumber)
                domain_opponent = domain + stringNumber + profileJsonOfOpponent
                domain_agent = domain + stringNumber + profileJsonOfAgent

                profile_parser_agent = ProfileParser()
                profile_parser_agent.parse(domain_agent)
                profile_parser_opponent = ProfileParser()
                profile_parser_opponent.parse(domain_opponent)

                domain_path = domain_path_1 + stringNumber + "/domain" + stringNumber + json_path

                with open(domain_path) as file:
                    domain_data = json.load(file)
                name = "domain_name"
                issue_values: Dict[str, ValueSet] = {}
                for issue_dict in domain_data['issuesValues'].keys():
                    mp: List[ImmutableList[Value]] = []
                    for value in domain_data['issuesValues'][issue_dict]['values']:
                        mp.append(cast(Value, value))
                    issue_values[issue_dict] = cast(ImmutableList[Value], mp)
                    # issue_values[issue_dict] = cast(Value,issue_values[issue_dict]['values'])

                domain_class = Domain(name, issue_values)
                """issues: List[Set[str]] = list(domain_class.getIssues())
                values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issues]
                all_bids: Outer = Outer[Value](values)"""
                for issue_dict in issue_values:
                    values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issue_values]
                    issue_values[issue_dict]: List[ImmutableList[Value]] = values
                all_bids_list = AllBidsList(domain_class)

                sorted_bids_agent = sorted(all_bids_list,
                                           key=lambda x: profile_parser_agent.getUtility_for_testing(x),
                                           reverse=True)
                sorted_bids_opponent = sorted(all_bids_list,
                                              key=lambda x: profile_parser_opponent.getUtility_for_testing(x),
                                              reverse=True)

                opponent_reg = AgentBrain(profile_parser_opponent, profile_parser_agent)
                agent_reg = AgentBrain(profile_parser_agent, profile_parser_opponent)

                agent_reg.fill_domain_and_profile(domain_class, profile_parser_agent, profile_parser_opponent)

                opponent_reg.fill_domain_and_profile(domain_class, profile_parser_opponent, profile_parser_agent)

                agent_reg.param = param
                opponent_reg.param = param

                bid_list_agent = []
                for m in range(1, bid_number_that_random):
                    if len(sorted_bids_opponent) > 5000:
                        bid_index = randint(0, int(50))
                        bid = sorted_bids_opponent[bid_index]
                        bid_list_agent.append(bid)
                        agent_reg.add_opponent_offer_to_self_x_and_self_y(bid, 0.1)
                    elif len(sorted_bids_opponent) > 1000:
                        bid_index = randint(0, int(20))
                        bid = sorted_bids_opponent[bid_index]
                        bid_list_agent.append(bid)
                        agent_reg.add_opponent_offer_to_self_x_and_self_y(bid, 0.1)
                    elif len(sorted_bids_opponent) > 20:
                        bid_index = randint(0, int(10))
                        bid = sorted_bids_opponent[bid_index]
                        bid_list_agent.append(bid)
                        agent_reg.add_opponent_offer_to_self_x_and_self_y(bid, 0.1)
                    else:
                        bid_index = randint(0, int(5))
                        bid = sorted_bids_opponent[bid_index]
                        bid_list_agent.append(bid)
                        agent_reg.add_opponent_offer_to_self_x_and_self_y(bid, 0.1)
                bid_list_opponent = []
                for m in range(1, bid_number_that_random):
                    if len(sorted_bids_agent) > 5000:
                        bid_index = randint(0, int(50))
                        bid = sorted_bids_agent[bid_index]
                        bid_list_opponent.append(bid)
                        opponent_reg.add_opponent_offer_to_self_x_and_self_y(bid, 0.1)
                    elif len(sorted_bids_agent) > 1000:
                        bid_index = randint(0, int(20))
                        bid = sorted_bids_agent[bid_index]
                        bid_list_opponent.append(bid)
                        opponent_reg.add_opponent_offer_to_self_x_and_self_y(bid, 0.1)
                    elif len(sorted_bids_agent) > 20:
                        bid_index = randint(0, int(10))
                        bid = sorted_bids_agent[bid_index]
                        bid_list_opponent.append(bid)
                        opponent_reg.add_opponent_offer_to_self_x_and_self_y(bid, 0.1)
                    else:
                        bid_index = randint(0, int(1))
                        bid = sorted_bids_opponent[bid_index]
                        bid_list_opponent.append(bid)
                        opponent_reg.add_opponent_offer_to_self_x_and_self_y(bid, 0.1)
                start = timer()

                agent_reg.param = param
                opponent_reg.param = param
                agent_reg.offers_unique = bid_list_agent
                opponent_reg.offers_unique = bid_list_opponent
                agent_reg.add_agent_first_n_bid_to_machine_learning_with_low_utility(sorted_bids_agent)
                opponent_reg.add_agent_first_n_bid_to_machine_learning_with_low_utility(sorted_bids_opponent)

                mae_agent = agent_reg.evaluate_data_according_to_lig_gbm(0.5)
                mae_opponent = opponent_reg.evaluate_data_according_to_lig_gbm(0.5)
                end = timer()
                print(end - start)

                average_mae_agent.append(float(mae_agent))
                average_mae_oppo.append(float(mae_opponent))
                dictionary = {'forEach': param, 'mae_agent': mae_agent, 'mae_opponent': mae_opponent,
                              'profile.agent': domain_agent,
                              'profile.oppo': domain_opponent,
                              'taken_random_bid_number_from_all_bid_list': bid_number_that_random,
                              'issue_weight': 2}
                """
                # Initialize an AutoML instance
                automl = LGBMRegressor()
                # Specify automl goal and constraint
                settings = {
                    "time_budget": 10,  # total running time in seconds
                    "metric": 'r2',  # primary metrics for regression can be chosen from: ['mae','mse','r2']
                    "estimator_list": ['lgbm'],  # list of ML learners; we tune lightgbm in this example
                    "task": 'regression',  # task type
                    "log_file_name": 'houses_experiment.log',  # flaml log file
                    "seed": 7654321,  # random seed
                }
                automl.fit(agent_reg.X, agent_reg.Y, None, settings)
                #automl.fit(X_train=agent_reg.X, y_train=agent_reg.Y, **settings)

                ypred = automl.predict(agent_reg.y_test)
                mae = mean_absolute_error(agent_reg.y_test, ypred)
                print('Best hyperparmeter config:', automl.best_config)
                print('Best r2 on validation data: {0:.4g}'.format(1 - automl.best_loss))
                """
                print(dictionary)

                result.append(dictionary)

            print("Average result for each parameter: ")
            print(param)
            avg_mea_agent_for_param = np.mean(average_mae_agent)
            avg_mea_oppo_for_param = np.mean(average_mae_oppo)
            dictionary2 = {'param': param, 'average_mae_agent': avg_mea_agent_for_param,
                           'average_mae_oppo': avg_mea_oppo_for_param}
            param_result.append(dictionary2)
    with open(f"result1.md", "w") as f:
        f.write(json.dumps(result))
    with open(f"param_result_1.md", "w") as f:
        f.write(json.dumps(param_result))


def corrolation_analyses():
    domain = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"
    domain_path_1 = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"
    profileJsonOfOpponent = "/profileA.json"
    profileJsonOfAgent = "/profileB.json"
    json_path = ".json"

    domain_analysis_dict = pd.DataFrame()

    value_dict = {}
    for k in range(0, 50):
        stringNumber = str(k).zfill(2)
        print(stringNumber)
        domain_opponent = domain + stringNumber + profileJsonOfOpponent
        domain_agent = domain + stringNumber + profileJsonOfAgent

        profile_parser_agent = ProfileParser()
        profile_parser_agent.parse(domain_agent)
        profile_parser_opponent = ProfileParser()
        profile_parser_opponent.parse(domain_opponent)

        domain_path = domain_path_1 + stringNumber + "/domain" + stringNumber + json_path

        with open(domain_path) as file:
            domain_data = json.load(file)
        name = "domain_name"
        issue_values: Dict[str, ValueSet] = {}
        for issue_dict in domain_data['issuesValues'].keys():
            mp: List[ImmutableList[Value]] = []
            for value in domain_data['issuesValues'][issue_dict]['values']:
                mp.append(cast(Value, value))
            issue_values[issue_dict] = cast(ImmutableList[Value], mp)
            # issue_values[issue_dict] = cast(Value,issue_values[issue_dict]['values'])

        domain_class = Domain(name, issue_values)
        """issues: List[Set[str]] = list(domain_class.getIssues())
        values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issues]
        all_bids: Outer = Outer[Value](values)"""
        for issue_dict in issue_values:
            values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issue_values]
            issue_values[issue_dict]: List[ImmutableList[Value]] = values
        all_bids_list = AllBidsList(domain_class)

        sorted_bids_agent = sorted(all_bids_list,
                                   key=lambda x: profile_parser_agent.getUtility_for_testing(x),
                                   reverse=True)
        sorted_bids_opponent = sorted(all_bids_list,
                                      key=lambda x: profile_parser_opponent.getUtility_for_testing(x),
                                      reverse=True)
        """
        opponent_reg = GradientBoostingRegressorModel(profile_parser_opponent, profile_parser_agent)
        agent_reg = GradientBoostingRegressorModel(profile_parser_agent, profile_parser_opponent)

        agent_reg.add_domain_and_profile(domain_class, profile_parser_agent, profile_parser_opponent)

        opponent_reg.add_domain_and_profile(domain_class, profile_parser_opponent, profile_parser_agent)

        agent_reg.param = param
        opponent_reg.param = param
        """
        storage_data = None
        try:
            with open("result1gamma.md") as file:
                storage_data = json.load(file)
            print("I load data from storage")
        except:
            print("Error skip")
        domian_spesific_data_list = []
        average_mae_agent = []
        average_mae_opponent = []
        for row in storage_data:
            if row['profile.agent'] == domain_agent:
                domian_spesific_data_list.append(row)
                average_mae_agent.append(float(row['mae_agent']))
                average_mae_opponent.append(float(row['mae_opponent']))

        average_list_agent = []
        average_list_opponent = []
        greater_than_095_agent = []
        greater_than_090_agent = []
        greater_than_085_agent = []
        greater_than_095_opponent = []
        greater_than_090_opponent = []
        greater_than_085_opponent = []
        for x in sorted_bids_agent:
            util = profile_parser_agent.getUtility_for_testing(x)
            average_list_agent.append(util)
            if float(util) > float(0.95):
                greater_than_095_agent.append(util)
            elif float(util) > float(0.90):
                greater_than_090_agent.append(util)
            elif float(util) > float(0.85):
                greater_than_085_agent.append(util)

        for x in sorted_bids_opponent:
            util = profile_parser_opponent.getUtility_for_testing(x)
            average_list_opponent.append(util)
            if float(util) > float(0.95):
                greater_than_095_opponent.append(util)
            elif float(util) > float(0.90):
                greater_than_090_opponent.append(util)
            elif float(util) > float(0.85):
                greater_than_085_opponent.append(util)
        percantage_of_greater_than95_agent = float(len(greater_than_095_agent)) / float(len(average_list_agent))
        percantage_of_greater_than90_agent = float(len(greater_than_090_agent)) / float(len(average_list_agent))
        percantage_of_greater_than85_agent = float(len(greater_than_085_agent)) / float(len(average_list_agent))
        percantage_of_greater_than95_opponent = float(len(greater_than_095_opponent)) / float(
            len(average_list_opponent))
        percantage_of_greater_than90_opponent = float(len(greater_than_090_opponent)) / float(
            len(average_list_opponent))
        percantage_of_greater_than85_opponent = float(len(greater_than_085_opponent)) / float(
            len(average_list_opponent))
        average = np.mean(average_list_agent)
        std = np.std(average_list_agent)
        list_size_agent = len(average_list_agent)
        list_size_oppo = len(average_list_opponent)
        value_dict['average'] = [average]
        value_dict['std'] = [std]
        value_dict['percantage_of_greater_than85_agent'] = [percantage_of_greater_than85_agent]
        value_dict['percantage_of_greater_than90_agent'] = [percantage_of_greater_than90_agent]
        value_dict['percantage_of_greater_than95_agent'] = [percantage_of_greater_than95_agent]
        value_dict['percantage_of_greater_than85_opponent'] = [percantage_of_greater_than85_opponent]
        value_dict['percantage_of_greater_than90_opponent'] = [percantage_of_greater_than90_opponent]
        value_dict['percantage_of_greater_than95_opponent'] = [percantage_of_greater_than95_opponent]
        value_dict['list_size_oppo'] = [list_size_oppo]
        value_dict['list_size_agent'] = [list_size_agent]
        value_dict['domain_opponent'] = [domain_opponent]
        value_dict['domain_agent'] = [domain_agent]
        value_dict['domain_res_agent'] = [domain_agent]
        value_dict['average_mae_agent'] = [np.mean(average_mae_agent)]
        value_dict['average_mae_opponent'] = [np.mean(average_mae_opponent)]
        df = pd.DataFrame.from_dict(value_dict)
        domain_analysis_dict = pd.concat([domain_analysis_dict, df])

    print("finished")
    corr = domain_analysis_dict.corr(method='kendall')
    print("corr")


def domain_analyses():
    domain = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"
    domain_path_1 = "C:/Users/nezih/Desktop/Anac2022/SunOfManAgent/domains/domain"
    profileJsonOfOpponent = "/profileA.json"
    profileJsonOfAgent = "/profileB.json"
    json_path = ".json"

    domain_analysis_dict = pd.DataFrame()

    value_dict = {}
    a = []
    for k in range(0, 50):
        stringNumber = str(k).zfill(2)
        print(stringNumber)
        domain_opponent = domain + stringNumber + profileJsonOfOpponent
        domain_agent = domain + stringNumber + profileJsonOfAgent

        profile_parser_agent = ProfileParser()
        profile_parser_agent.parse(domain_agent)
        profile_parser_opponent = ProfileParser()
        profile_parser_opponent.parse(domain_opponent)

        domain_path = domain_path_1 + stringNumber + "/domain" + stringNumber + json_path

        with open(domain_path) as file:
            domain_data = json.load(file)
        name = "domain_name"
        issue_values: Dict[str, ValueSet] = {}
        for issue_dict in domain_data['issuesValues'].keys():
            mp: List[ImmutableList[Value]] = []
            for value in domain_data['issuesValues'][issue_dict]['values']:
                mp.append(cast(Value, value))
            issue_values[issue_dict] = cast(ImmutableList[Value], mp)
            # issue_values[issue_dict] = cast(Value,issue_values[issue_dict]['values'])

        domain_class = Domain(name, issue_values)
        """issues: List[Set[str]] = list(domain_class.getIssues())
        values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issues]
        all_bids: Outer = Outer[Value](values)"""
        for issue_dict in issue_values:
            values: List[ImmutableList[Value]] = [domain_class.getValues(issue) for issue in issue_values]
            issue_values[issue_dict]: List[ImmutableList[Value]] = values
        all_bids_list = AllBidsList(domain_class)

        sorted_nash = sorted(all_bids_list,
                             key=lambda x: profile_parser_agent.getUtility_for_testing(
                                 x) + profile_parser_opponent.getUtility_for_testing(x),
                             reverse=True)
        b = profile_parser_agent.getUtility_for_testing(
            sorted_nash[0]) + profile_parser_opponent.getUtility_for_testing(sorted_nash[0])
        a.append(b)
        value_dict['agent_opponnet_max_sum_util'] = float(b)
        average_list_agent = []
        average_list_opponent = []
        greater_than_095_agent = []
        greater_than_090_agent = []
        greater_than_085_agent = []
        greater_than_095_opponent = []
        greater_than_090_opponent = []
        greater_than_085_opponent = []
        sorted_bids_agent = sorted(all_bids_list,
                                   key=lambda x: profile_parser_agent.getUtility_for_testing(x),
                                   reverse=True)
        sorted_bids_opponent = sorted(all_bids_list,
                                      key=lambda x: profile_parser_opponent.getUtility_for_testing(x),
                                      reverse=True)
        for x in sorted_bids_agent:
            util = profile_parser_agent.getUtility_for_testing(x)
            average_list_agent.append(util)
            if float(util) > float(0.95):
                greater_than_095_agent.append(util)
            elif float(util) > float(0.90):
                greater_than_090_agent.append(util)
            elif float(util) > float(0.85):
                greater_than_085_agent.append(util)

        for x in sorted_bids_opponent:
            util = profile_parser_opponent.getUtility_for_testing(x)
            average_list_opponent.append(util)
            if float(util) > float(0.95):
                greater_than_095_opponent.append(util)
            elif float(util) > float(0.90):
                greater_than_090_opponent.append(util)
            elif float(util) > float(0.85):
                greater_than_085_opponent.append(util)
        percantage_of_greater_than95_agent = float(len(greater_than_095_agent)) / float(len(average_list_agent))
        percantage_of_greater_than90_agent = float(len(greater_than_090_agent)) / float(len(average_list_agent))
        percantage_of_greater_than85_agent = float(len(greater_than_085_agent)) / float(len(average_list_agent))
        percantage_of_greater_than95_opponent = float(len(greater_than_095_opponent)) / float(
            len(average_list_opponent))
        percantage_of_greater_than90_opponent = float(len(greater_than_090_opponent)) / float(
            len(average_list_opponent))
        percantage_of_greater_than85_opponent = float(len(greater_than_085_opponent)) / float(
            len(average_list_opponent))
        average = np.mean(average_list_agent)
        std = np.std(average_list_agent)
        list_size_agent = len(average_list_agent)
        list_size_oppo = len(average_list_opponent)
        value_dict['average'] = [average]
        value_dict['std'] = [std]
        value_dict['percantage_of_greater_than85_agent'] = [percantage_of_greater_than85_agent]
        value_dict['percantage_of_greater_than90_agent'] = [percantage_of_greater_than90_agent]
        value_dict['percantage_of_greater_than95_agent'] = [percantage_of_greater_than95_agent]
        value_dict['percantage_of_greater_than85_opponent'] = [percantage_of_greater_than85_opponent]
        value_dict['percantage_of_greater_than90_opponent'] = [percantage_of_greater_than90_opponent]
        value_dict['percantage_of_greater_than95_opponent'] = [percantage_of_greater_than95_opponent]
        value_dict['list_size_oppo'] = [list_size_oppo]
        value_dict['list_size_agent'] = [list_size_agent]
        value_dict['domain_opponent'] = [domain_opponent]
        value_dict['domain_agent'] = [domain_agent]
        value_dict['domain_res_agent'] = [domain_agent]

        df = pd.DataFrame.from_dict(value_dict)
        domain_analysis_dict = pd.concat([domain_analysis_dict, df])

    print("mean: " + str(np.mean(a)) + "max + " + str(np.max(a)) + "min: " + str(np.min(a)))

    corr = domain_analysis_dict.corr(method="kendall")
    print("cor")


"""
if __name__ == "__main__":
    if True:
        deep_analysis_for_machine_learning()
    if False:
        corrolation_analyses()
    if False:
        domain_analyses()
"""
