# archlearn-mini
ArchLearn Mini is a small-scale information retrieval and text-analytics system designed to map and visualize research patterns surrounding creative learning in architecture and art education.

The system is built as an web app which can be accessed at: https://archlearnminiv1-zeshanasif.streamlit.app/
(User might need to wake the app up before using, takes 2 mins and a reload)

There are 2 modes of IR search. A simple mode (uses TF-IDF). The source code can be found in the file titled "search_cli.py"
The advanced search uses a combination of TF-IDF, BM25 and semantic scores. The source code can be found at "search_advanced.py" file

If the app needs to run on a local machine, just run the compiled version on the file called "appV2.py"
