import asyncio
import json
import time
from pathlib import Path

from utils.plot_trace import plot_trace
from utils.runners import run_session


def main():
    RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

    # create results directory if it does not exist
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    # Settings to run a negotiation session:
    #   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
    #   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
    #   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement
    domain = "domains/domain"
    profileJsonOfOpponent = "/profileA.json"
    profileJsonOfAgent = "/profileB.json"
    for i in [3, 4, 5, 7]:
        stringNumber = str(i).zfill(2)
        print(stringNumber)
        settings = {
            "agents": [
                {
                    "class": "agents.SUN_AGENT.SUN_Agent.SunAgent",
                    "parameters": {"storage_dir": "agent_storage/SunAgent"},
                },
                {
                    "class": "agents.CSE3210.agent26.agent26.Agent26",
                }
            ],
            "profiles": [domain + stringNumber + profileJsonOfOpponent, domain + stringNumber + profileJsonOfAgent],
            "deadline_time_ms": 60000,
        }

        # run a session and obtain results in dictionaries
        session_results_trace, session_results_summary = run_session(settings)
        # plot trace to html file
        if not session_results_trace["error"]:
            plot_trace(session_results_trace, RESULTS_DIR.joinpath("trace_plot" + str(stringNumber) + ".html"))

        # write results to file
        with open(RESULTS_DIR.joinpath("session_results_trace+" + str(stringNumber) + ".json"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(session_results_trace, indent=2))
        with open(RESULTS_DIR.joinpath("session_results_summary" + str(stringNumber) + ".json"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(session_results_summary, indent=2))


if __name__ == "__main__":
    main()
