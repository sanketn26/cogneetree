"""Verification script for Cogneetree storage backends."""

import os
import shutil
from cogneetree import ContextWorkflow, Config
from cogneetree.storage import InMemoryStorage, JsonFileStorage, MarkdownFileStorage, SQLiteStorage

def test_storage(name, storage):
    print(f"\n--- Testing {name} ---")
    workflow = ContextWorkflow(storage=storage)
    
    try:
        with workflow.session("s1", "Test Ask", "Test Plan") as session:
            print(f"Session created: {session.manager.get_current_session().session_id}")
            with session.activity("a1", "Test Activity", "test", "core", "analysis") as activity:
                with activity.task("Test Task", ["tag1", "tag2"]) as task:
                    task.record_action("Action 1")
                    task.set_result("Result 1")
        
        # Verify persistence (read back)
        if hasattr(storage, "get_session"):
            s = storage.get_session("s1")
            if s:
                print(f"✅ Verified read back: {s.session_id}")
            else:
                print("❌ Failed to read back session")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    # Cleanup
    if os.path.exists(".test_data"):
        shutil.rmtree(".test_data")
    os.makedirs(".test_data")

    # 1. In-Memory
    test_storage("InMemory", InMemoryStorage())

    # 2. JSON File
    test_storage("JSON File", JsonFileStorage(".test_data/data.json"))

    # 3. Markdown File
    test_storage("Markdown File", MarkdownFileStorage(".test_data/log.md"))

    # 4. SQLite
    test_storage("SQLite", SQLiteStorage(".test_data/db.sqlite"))

    print("\nCheck .test_data/ for output files.")

if __name__ == "__main__":
    main()
