import gradio as gr
import os
import subprocess
from pathlib import Path
import sys

from backend.embeddings_manager import get_embedding_model, MODELS_DICT
from backend.vector_db_manager import VectorDbManager
from backend.inference import InferenceInstance, read_relevant_content
from backend.pdf_to_mmd import pdf_to_mmd
from backend.logger import Logger, read_logs

import time


def get_accessible_port():
    from socket import socket

    with socket() as s:
        s.bind(('', 0))
        return int(s.getsockname()[1])


port = get_accessible_port()


# Launch a simple HTTP server to serve the PDF files
def start_server():
    command = ['python', '-m', 'http.server', f"{port}"]
    # Set the working directory to the documents folder to serve the PDF files
    os.chdir(Path(os.getcwd()).parent / "documents")
    subprocess.Popen(command)
    # Return to the original working directory
    os.chdir(Path(os.getcwd()).parent / "front_end")


# Start the server
start_server()

# Global variable etc...

BASE_DB_DIRECTORY = Path(r"../documents/vector_db")


def update_embedding(embedding_name):
    global BASE_DB_DIRECTORY, VECTOR_DB_MANAGER, INFERENCE_INSTANCE
    embedding_func = get_embedding_model(embedding_name)
    VECTOR_DB_MANAGER = VectorDbManager(embedding_name=embedding_name, embedding_function=embedding_func, chunk_size=512, db_directory=BASE_DB_DIRECTORY)
    INFERENCE_INSTANCE = InferenceInstance(vector_db_manager=VECTOR_DB_MANAGER, nb_chunks_retrieved=4)
    print(f"Updated embedding model to {embedding_name}")


update_embedding("intfloat/multilingual-e5-large")
user_message_global = ""


def user(user_message, history):
    global user_message_global
    user_message_global = user_message
    return "", history + [[user_message, None]]


def bot(history):
    global user_message_global, doc_path

    if doc_path != "":
        print(f"FOUND DOC_PATH {doc_path}")
        doc_extension = doc_path.split(".")[-1]
        if doc_extension == "mmd":
            VECTOR_DB_MANAGER.create_vector_store_from_latex(Path(doc_path))
        elif doc_extension == "pdf":
            VECTOR_DB_MANAGER.create_vector_store_from_pdf(doc_path)
        else:
            print(f"Unsupported extension: {doc_extension}")
    else:
        print("NOT FOUND DOC_PATH")

    doc_name = Path(doc_path).stem + ".mmd" if math_checkbox.value else Path(doc_path).name
    bot_message = INFERENCE_INSTANCE.get_next_token(user_message_global, doc_name)
    history[-1][1] = ""
    for message in bot_message:
        history[-1][1] = message
        time.sleep(0.05)
        yield history


def update_path(p, checked):
    """Update the global variable doc_path with the selected PDF path"""
    print("Updating path")
    global doc_path
    name = Path(p).name
    stem = Path(p).stem
    if checked:
        if not (Path(r"../documents/mmds") / (stem + ".mmd")).exists():
            pdf_to_mmd(r"../documents/pdfs/" + name)
        print(f"Selected DOC path: {stem}.mmd")
        doc_path = r"../documents/mmds/" + stem + ".mmd"
    else:
        print(f"Selected DOC path: {name}")
        doc_path = str(p)


def pdf_viewer(pdf_file):
    """Display the PDF file in an HTML viewer"""
    pdf_path = Path(pdf_file)
    pdf_working_dir = Path(os.getcwd()).parent / "documents" / "pdfs"

    # Check if the PDF file is in the working directory
    if not (pdf_working_dir / pdf_path.name).exists():
        return f"""<h1>File {pdf_path.name} not found in the working directory</h1>
                   <p>You can only access PDFs that are inside {pdf_working_dir}</p>"""

    # Create the HTML code for the PDF viewer
    return f"""
    <iframe
        src="http://localhost:{port}/pdfs/{pdf_path.name}"
        width="100%"
        height="800px"
        style="border:none;"
    ></iframe>
    """


# Define main Gradio tab
with gr.Blocks() as main_tab:
    with gr.Row():
        with gr.Column(scale = 3):
            with gr.Row():
                with gr.Column(scale=12):
                    pdf_output = gr.HTML()
            with gr.Row():
                with gr.Column(scale=12):
                    file_input = gr.File(label="Select a PDF file")
                    math_checkbox = gr.Checkbox(label="Interpret as LaTeX (a latex version will be created then given to "
                                                      "the chatbot, the conversion take some time)")

            with gr.Group():
                chatbot = gr.Chatbot(scale=2,
                                     latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                       {"left": "$", "right": "$", "display": False}])
                msg = gr.Textbox(label="User message", scale=2)

                msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot, chatbot, chatbot
                )



        with gr.Column():
            references = gr.Markdown(label="References",
                                     latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                       {"left": "$", "right": "$", "display": False}])
            main_tab.load(read_relevant_content, None, references, every=1)

    file_input.change(pdf_viewer, inputs=file_input, outputs=pdf_output)
    file_input.upload(update_path, inputs=[file_input, math_checkbox])


# Define the log tab
with gr.Blocks() as log_tab:
    logs = gr.Textbox(lines=50, interactive=False)
    sys.stdout = Logger("../temp_file/output.log")
    log_tab.load(read_logs, None, logs, every=1)


# Define options tab
with gr.Blocks() as options_tab:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=12):
                # TODO: Add options for the inference instance
                gr.Textbox(label="Options", scale=2)
                embedding_model_dropdown = gr.Dropdown(label="Embedding model", choices=MODELS_DICT.keys(), value="intfloat/multilingual-e5-large")
                embedding_model_dropdown.change(update_embedding, inputs=embedding_model_dropdown)


# Define conversion tab
with gr.Blocks() as conversion_tab:
    with gr.Column():
        file_input = gr.File(label="Select a PDF file to convert to MMD")
        html_output = gr.HTML(label="Output")



    def upload_func(file_input):
        name = Path(file_input).name
        file_path = fr"../documents/pdfs/{name}"
        pdf_to_mmd(file_path)


    file_input.upload(upload_func, inputs=file_input)


app = gr.TabbedInterface([main_tab, log_tab, options_tab, conversion_tab],
                         ["Main", "Logs", "Options", "Conversion"])
app.queue()
app.launch()

