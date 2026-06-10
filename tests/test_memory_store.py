import tempfile
import unittest
from pathlib import Path

from agent.memory_store import MemoryStore


class MemoryStoreTests(unittest.TestCase):

    def test_saves_and_loads_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"

            store = MemoryStore(memory_path)
            store.set_profile("Name", "Sam")
            store.set_preference("Favorite Color", "green")
            self.assertTrue(store.add_fact("User is learning Python."))

            reloaded = MemoryStore(memory_path)

            self.assertEqual(reloaded.get_profile()["name"], "Sam")
            self.assertEqual(
                reloaded.get_preferences()["favorite_color"],
                "green",
            )
            self.assertEqual(
                reloaded.get_facts()[0]["text"],
                "User is learning Python.",
            )

    def test_duplicate_facts_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Path(temp_dir) / "memory.json")

            self.assertTrue(store.add_fact("User likes short answers."))
            self.assertFalse(store.add_fact("user likes short answers."))
            self.assertEqual(len(store.get_facts()), 1)

    def test_prompt_context_contains_memory_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(Path(temp_dir) / "memory.json")
            store.set_profile("name", "Sam")
            store.set_preference("tone", "concise")
            store.add_fact("User is testing memory.")

            context = store.to_prompt_context()

            self.assertIn("User profile: name: Sam", context)
            self.assertIn("User preferences: tone: concise", context)
            self.assertIn("- User is testing memory.", context)


if __name__ == "__main__":
    unittest.main()
