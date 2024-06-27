##### Data Science Assignment

# FAQ chatbot using Retrieval Augmented Generation.

Objective: To create an AI chatbot for customers that can answer FAQ questions.

### Objective:

The goal of our project is to build a web based chatbot that can answer frequently asked questions from our customers.
The chatbot should:

1. Use Retrieval Augmented Generation
2. Chat History as Context

What is RAG?

Retrieval Augmented Generation (RAG) is a process used in Language Learning Models (LLMs) to incorporate external data into the generation step. It involves retrieving relevant data from external sources and using it during the generation process as a part of the prompt.

It involves:

1. Matching the user question with documents in a document store also called vector store.
2. Stuffing the top n matches into the prompt as relavant context.
3. Answering the original question taking into consideration only the retrieved context.

**Steps Involved**

1. Load PDF using document loader.
2. Splitting Text into Each question called chunks.
3. Storing chunks into a Vector Database in the form of Embeddings.
4. Creating a retrieval Question-Answer chain by retrieving the context from the vector database
5. Creating a conversation retrieval Question-Answer chain using chat history
