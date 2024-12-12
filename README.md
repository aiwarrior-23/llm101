# LLM for All

This article series is for Generative AI enthusiasts to learn LLM modeling from start using three frameworks:
1. OpenAI
2. LangChain
3. Llama-index

This is a full hands-on series with UI integration. The UI is built using React JS. Given below is the current list of articles:

## Table of Contents
1. [Building your own GPT](#article-1-building-your-own-gpt)
2. [Adding Conversational Memory to LLM Chatbot](#article-2-adding-conversational-memory-to-llm-chatbot)

## Article 1: Building your own GPT

1. **OpenAI API Integration**: Set up a basic ChatGPT model using OpenAI’s API, offering a strong foundational conversational experience.
2. **LangChain Integration**: Enhances ChatGPT's capabilities by integrating LangChain, allowing for more complex workflows and customization in responses.
3. **Llama Index Integration**: Utilizes Llama Index to construct an alternative GPT-based solution, providing an additional layer of flexibility and a different approach to response generation.
4. **UI Integration**: Connects models to a React frontend, enabling real-time interaction through a user-friendly chat interface.
5. **Configurable Settings**: Users can customize settings directly from the UI, making it easier to adjust the model’s behavior as needed without diving into the backend.

**HOW TO RUN THE CODE:**


Rename App_session1.js to App.js and run the below command


`uvicorn routes_session1:app --reload` from backend dir


`npm start` from frontend\llm_ui dir


For more details on how to implement and deploy this setup, check out the complete guide on [Medium](https://medium.com/@himanshuit3036/llm-for-all-1-building-your-own-gpt-17dd3d9701dc).

## Article 2: Adding Conversational Memory to LLM Chatbot

1. **Conversational Memory with OpenAI API**: Adds conversational memory to the chatbot by tracking the message history, allowing the model to recall past interactions. This history is stored using Redis for persistence.
2. **Conversational Memory with LangChain**: Uses LangChain’s `ConversationSummaryMemory()` to summarize past conversations, maintaining a manageable context window to optimize token usage.
3. **Conversational Memory with Llama Index**: Implements Llama Index’s `ChatSummaryMemoryBuffer()` for summarizing memory, providing an alternative way to handle conversation history.
4. **Enhanced Chatbot UI**: Updates the UI structure to resemble a typical chatbot interface, making it more intuitive for ongoing interactions.
5. **Cloud Deployment**: Deploys the chatbot with memory capabilities to AWS using Amplify, enabling accessible and scalable usage.

**HOW TO RUN THE CODE:**


Rename App_session2.js to App.js and run the below command


`uvicorn routes_session2:app --reload` from backend dir


`npm start` from frontend\llm_ui dir


Make sure to rename to App_session1.jsx from previous App.js

For more details on how to implement and deploy this setup, check out the complete guide on [Medium](https://medium.com/@himanshuit3036/llm-for-all-1-building-your-own-gpt-17dd3d9701dc).

## Article 3: Adding Conversational Memory to LLM Chatbot

1. **Creating a Redis Instance**: Creation of Redis Docker Instance for session and chat storage.
2. **Persisting conversations in Open AI Framework**: Using Open AI Framework to converse with LLM and save chat history.
3. **Persisting conversations in LangChain Framework**: Using LangChain Framework to converse with LLM and save chat history.
4. **Persisting conversations in Llama Index Framework**: Using Llama Index Framework to converse with LLM and save chat history.
5. **Retrieving sessions and chat histories**: Retrieve sessions and chat histories based on Framework selected.
6. **Creation of Docker Compose**: Containerization of entire application using Docker
7. **Creating AWS services using CloudFormation**: Deployment in AWS using CloudFromation Template
8. **Automation using GitHub Actions**: Automating CICD using GitHub Actions

**HOW TO RUN THE CODE:**


Rename routes_session3.py to routes.py and run the below command


`docker-compose up --build`


Make sure to rename to App_session3.jsx from previous App.js as well

For more details on how to implement and deploy this setup, check out the complete guide on [Medium](https://medium.com/@himanshuit3036/llm-for-all-03-gpt-powered-chatbot-with-redis-cache-and-aws-deployment-with-ci-cd-a0b7d2d9a2f9).
