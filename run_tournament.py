import json
import os
from pathlib import Path
import time

from utils.runners import run_tournament

RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

# create results directory if it does not exist
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

# Settings to run a negotiation session:
#   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
#   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
#   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement.
tournament_settings = {
    "agents": [
        {
            "class": "agents.SUN_AGENT.SUN_Agent.SunAgent",
            "parameters": {"storage_dir": "agent_storage/SunAgent"},
        },
        {
            "class": "agents.CSE3210.agent26.agent26.Agent26",
        },
        {
            "class": "agents.CSE3210.agent64.agent64.Agent64",
        },
        {
            "class": "agents.CSE3210.agent33.agent33.Agent33",
        },
        {
            "class": "agents.CSE3210.agent14.agent14.Agent14",
        },
        {
            "class": "agents.CSE3210.agent18.agent18.Agent18",
        },
        {
            "class": "agents.CSE3210.agent19.agent19.Agent19",
        },
        {
            "class": "agents.CSE3210.agent22.agent22.Agent22",
        },
        {
            "class": "agents.CSE3210.agent24.agent24.Agent24",
        },
        {
            "class": "agents.CSE3210.agent25.agent25.Agent25",
        }
    ],
    "profile_sets": [
        ["domains/domain03/profileA.json", "domains/domain03/profileB.json"],
        ["domains/domain04/profileA.json", "domains/domain04/profileB.json"],
        ["domains/domain05/profileA.json", "domains/domain05/profileB.json"],
        ["domains/domain07/profileA.json", "domains/domain07/profileB.json"],
        ["domains/domain18/profileA.json", "domains/domain18/profileB.json"],
        ["domains/domain19/profileA.json", "domains/domain19/profileB.json"],
        ["domains/domain20/profileA.json", "domains/domain20/profileB.json"],
        ["domains/domain21/profileA.json", "domains/domain21/profileB.json"],
        ["domains/domain22/profileA.json", "domains/domain22/profileB.json"],
        ["domains/domain23/profileA.json", "domains/domain23/profileB.json"],
        ["domains/domain24/profileA.json", "domains/domain24/profileB.json"],
        ["domains/domain28/profileA.json", "domains/domain28/profileB.json"],
        ["domains/domain30/profileA.json", "domains/domain30/profileB.json"],
        ["domains/domain47/profileA.json", "domains/domain47/profileB.json"],
        ["domains/domain48/profileA.json", "domains/domain48/profileB.json"],
        ["domains/domain49/profileA.json", "domains/domain49/profileB.json"]
    ],
    "deadline_time_ms": 30000,
    # "deadlinerounds":{"rounds":100,"durationms":999}
}

# run a session and obtain results in dictionaries
tournament_steps, tournament_results, tournament_results_summary = run_tournament(tournament_settings)

# save the tournament settings for reference
with open(RESULTS_DIR.joinpath("tournament_steps.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(tournament_steps, indent=2))
# save the tournament results
with open(RESULTS_DIR.joinpath("tournament_results.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(tournament_results, indent=2))
# save the tournament results summary
tournament_results_summary.to_csv(RESULTS_DIR.joinpath("tournament_results_summary.csv"))
