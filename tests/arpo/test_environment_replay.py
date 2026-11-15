"""Use local ALFWorld data to verify replay without Ray or model inference."""
from pathlib import Path
import os
import unittest

from agent_system.environments.env_package.alfworld.envs import AlfworldWorker, load_config_file
from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv


class EnvironmentReplayTests(unittest.TestCase):
    def test_real_game_fork_preserves_state_and_next_game_schedule(self):
        data = Path(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")).expanduser()
        if not (data / "json_2.1.1/train").is_dir():
            self.skipTest("Local ALFWorld training data is unavailable")
        gamefiles = []
        for game in (data / "json_2.1.1/train").rglob("game.tw-pddl"):
            gamefiles.append(str(game))
            if len(gamefiles) == 2:
                break
        if len(gamefiles) < 2:
            self.skipTest("Need at least two ALFWorld games")
        config_path = Path(__file__).resolve().parents[2] / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
        config = load_config_file(str(config_path))
        base = AlfredTWEnv.__new__(AlfredTWEnv)
        base.config, base.train_eval, base.use_expert = config, "train", False
        base.game_files, base.num_games = gamefiles, len(gamefiles)
        source = AlfworldWorker(config, 7, base)
        target = AlfworldWorker(config, 7, base)
        try:
            source.reset()
            target.reset()
            source.step("look")
            state = source.snapshot()
            target.restore(state)
            self.assertEqual(target.snapshot(), state)
            # Follow with another action, so equal text alone cannot mask an
            # incorrectly restored simulator's transition behavior.
            self.assertEqual(source.step("inventory"), target.step("inventory"))
            # Replay must not consume an extra game from the target's iterator.
            self.assertEqual(source.reset(), target.reset())
        finally:
            source.env.close()
            target.env.close()


if __name__ == "__main__":
    unittest.main()
