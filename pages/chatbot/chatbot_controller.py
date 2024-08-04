from dash.dependencies import Input, Output, State
from app import app

from components.textbox import render_textbox
from pages.chatbot.chatbot_model import conversation_chain, vectordb, MODEL_NAME

@app.callback(
    Output(component_id="display-conversation", component_property="children"), 
    Input(component_id="store-conversation", component_property="data")
)
def update_display(chat_history):
    return [
        render_textbox(x, box="human") if i % 2 == 0 else render_textbox(x, box="AI")
        for i, x in enumerate(chat_history.split("<split>")[:-1])
    ]

@app.callback(
    Output(component_id="user-input", component_property="value"),
    Input(component_id="submit", component_property="n_clicks"), 
    Input(component_id="user-input", component_property="n_submit"),
)
def clear_input(n_clicks, n_submit):
    return ""

@app.callback(
    Output(component_id="store-conversation", component_property="data"), 
    Output(component_id="loading-component", component_property="children"),
    Input(component_id="submit", component_property="n_clicks"), 
    Input(component_id="user-input", component_property="n_submit"),
    State(component_id="user-input", component_property="value"), 
    State(component_id="store-conversation", component_property="data"),
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    if n_clicks == 0 and n_submit is None:
        return "", None

    if user_input is None or user_input == "":
        return chat_history, None
    
    chat_history += f"Human: {user_input}<split>ChatBot: "

    if MODEL_NAME == "GPT":
        output = conversation_chain.invoke(user_input)
    else:
        print('Retrieving Similar Documents..')
        docs = vectordb.similarity_search(user_input)

        print('Getting output from the LLM model...')
        inputs = {"input_documents": docs, "question": user_input}  
        output = conversation_chain.run(inputs)

    chat_history += f"{output}<split>"

    # chat_history += "response<split>"
    return chat_history, None