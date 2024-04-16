import gradio as gr
import os
import subprocess
from pathlib import Path

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from backend.vector_db_manager import VectorDbManager
from backend.inference import InferenceInstance
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

# Create VectorDbManager and Inference instance

embedding_func = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs={'device': 'cuda'})
base_db_directory = Path(r"../documents/vector_db")
vector_db_manager = VectorDbManager(embedding_name="multilingual-e5-large", embedding_function=embedding_func, chunk_size=512, db_directory=base_db_directory)
inference_instance = InferenceInstance(vector_db_manager=vector_db_manager, nb_chunks_retrieved=4)


user_message_global = ""


def user(user_message, history):
    global user_message_global
    user_message_global = user_message
    return "", history + [[user_message, None]]


def bot(history):
    global user_message_global, doc_path

    if doc_path != "":
        print("FOUND DOC_PATH")
        vector_db_manager.create_vector_store_from_pdf(doc_path)
    else:
        print("NOT FOUND DOC_PATH")

    bot_message = inference_instance.get_next_token(user_message_global, doc_path.split("\\")[-1])
    history[-1][1] = ""
    for message in bot_message:
        history[-1][1] = message
        time.sleep(0.05)
        yield history


def update_path(p):
    """Update the global variable doc_path with the selected PDF path"""
    global doc_path
    doc_path = str(p)
    print(f"Selected PDF path: {doc_path}")


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
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=12):
                pdf_output = gr.HTML()
        with gr.Row():
            with gr.Column(scale=12):
                file_input = gr.File(label="Select a PDF file")

    with gr.Column():
        with gr.Group():
            chatbot = gr.Chatbot(scale=2)
            msg = gr.Textbox(scale=2)
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )

    file_input.change(pdf_viewer, inputs=file_input, outputs=pdf_output)
    file_input.upload(update_path, inputs=file_input)


# Define options tab
with gr.Blocks() as options_tab:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=12):
                # TODO: Add options for the inference instance
                gr.Textbox(label="Options", scale=2)


app = gr.TabbedInterface([main_tab, options_tab], ["Main", "Options"])
app.queue()
app.launch()
