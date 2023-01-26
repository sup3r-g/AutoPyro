import sys
from streamlit.web import cli as stcli

def interactive_session():
    sys.argv = ["streamlit", "run", "home.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    interactive_session()
