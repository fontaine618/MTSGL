init:
    pip install -r requirements.txt

test:
    unittest discover tests

.PHONY: init test