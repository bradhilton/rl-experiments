from dotenv import load_dotenv
import os

load_dotenv()

if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    )
