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

For more details on how to implement and deploy this setup, check out the complete guide on [Medium](https://medium.com/@himanshuit3036/llm-for-all-1-building-your-own-gpt-17dd3d9701dc).

## Article 2: Adding Conversational Memory to LLM Chatbot

1. **Conversational Memory with OpenAI API**: Adds conversational memory to the chatbot by tracking the message history, allowing the model to recall past interactions. This history is stored using Redis for persistence.
2. **Conversational Memory with LangChain**: Uses LangChain’s `ConversationSummaryMemory()` to summarize past conversations, maintaining a manageable context window to optimize token usage.
3. **Conversational Memory with Llama Index**: Implements Llama Index’s `ChatSummaryMemoryBuffer()` for summarizing memory, providing an alternative way to handle conversation history.
4. **Enhanced Chatbot UI**: Updates the UI structure to resemble a typical chatbot interface, making it more intuitive for ongoing interactions.
5. **Cloud Deployment**: Deploys the chatbot with memory capabilities to AWS using Amplify, enabling accessible and scalable usage.

For more details on how to implement and deploy this setup, check out the complete guide on [Medium](https://medium.com/@himanshuit3036/llm-for-all-1-building-your-own-gpt-17dd3d9701dc).
