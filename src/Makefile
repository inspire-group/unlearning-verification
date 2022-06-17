VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip


run: $(VENV)/bin/activate
	cd single_user; \
	../$(PYTHON) generate_paper_data.py --emnist -r 5


$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install keras==2.6.0 tensorflow==2.6.0  # please do not ask.. compatibility issues
	$(PIP) install -r requirements.txt 


clean:
	rm -rf __pycache__
	rm -rf $(VENV)