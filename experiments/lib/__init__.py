from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
)
