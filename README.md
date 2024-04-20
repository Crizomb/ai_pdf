Chat locally with any PDF

Ask questions, get answer with usefull references 

Work well with math pdfs (convert them to LaTex, a math syntax comprehensible by computer)

## Work flow chart

![RAG_diagrams](https://github.com/Crizomb/ai_pdf/assets/62544756/430c28ac-ed48-4ac5-99ae-58b7f810250c)


## Demos

chatbot test with some US Laws pdf

https://github.com/Crizomb/ai_pdf/assets/62544756/b399d5bc-df2f-4be0-b6fe-0c272f915c72

chatbot test with math pdf (interpereted as latex by the LLM)

https://github.com/Crizomb/ai_pdf/assets/62544756/eebf5520-bf78-4b82-8699-782e6d7147c4

full length process of converting pdf to latex, then using the chat bot

https://github.com/Crizomb/ai_pdf/assets/62544756/a10238f1-2e26-4a97-94d0-d32ec52ee195




## How to use 

Clone the project to some location that we will call 'x'
install requierements listed in the requirements.txt file
(open terminal, go to the 'x' location, run pip install -r requirements.txt)
([OPTIONAL] for better performance during embedding, install pytorch with cuda, go to https://pytorch.org/get-started/locally/) 

Put your pdfs in x/ai_pdf/documents/pdfs
Run x/ai_pdf/main.py
Select or not math mode
Choose the pdf you want to work on
Wait a little bit for the pdf to get vectorized (check task manager to see if your gpu is going vrum)

Launch LM Studio, Go to the local Server tab, choose the model you want to run, choose 1234 as server port, start server
(If you want to use open-ai or any other cloud LLM services, change line 10 of x/ai_pdf/back_end/inference.py with your api_key and your provider url)

Ask questions to the chatbot
Get answer
Go eat cookies


### TODO 

- [ ] Option tabs
    - [ ] add more different embedding models
    - [ ] add menu to choose how many relevant chunk of information the vector search should get from the vector db
    - [ ] menu to configure api url and api key
     
## Maybe in the futur

- [ ] Add special support for code PDF (with specialized langchain code spliter)
- [ ] Add Multimodality
      

