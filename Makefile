run_black:
	python -m black . -l 119

run_server:
	uvicorn app.backend.backend:app --reload

run_client:
	python -m streamlit run app/frontend/frontend.py --browser.serverAddress localhost 

run_app: run_server run_client
